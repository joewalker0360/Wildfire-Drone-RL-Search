"""
interactive_game.py
Full-Featured Interactive Wildfire Drone Rescue Game & Human-in-the-Loop RL Studio.
Run with:
    python interactive_game.py

Rules:
1. Blind AI Benchmark: The RL model scouts the procedural maze first, solves it, and sets the hidden step limit.
2. No Paths or Step Limits Shown: Neither the AI path nor the step limit is revealed during flight!
   The pilot must predict the shortest, safest route on their own.
3. 2-Step Allowance: The player receives a +2 step allowance over the RL model's benchmark.
4. Level Failure: If the player exceeds the hidden limit (RL steps + 2), the level fails, reveals the comparison,
   and restarts THAT EXACT LEVEL from the beginning.
5. Level Success: If the player reaches the hospital with the ranger within the limit, the level advances,
   reveals the comparison, generates a brand-new maze, and the RL model scouts the next level.
"""

import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dynamic_wildfire_env import DynamicWildfireDroneEnv
from interactive_learning import InteractiveQLearningAgent
from wildfire_model_manager import WildfireModelManager

PORT = 8000
HOST = "127.0.0.1"


class GameEngine:
    def __init__(self):
        self.env = DynamicWildfireDroneEnv(reward_mode="risk_shaped", auto_randomize_on_success=True)
        self.manager = WildfireModelManager(self.env)
        self.agent = self.manager.interactive_q_agent
        self.current_trajectory = []
        self.last_action = None
        self.human_demonstrations = []
        self.auto_randomize_on_success = True
        
        # RL Model plays first in secret to benchmark initial level
        self.rl_scout_result = self.manager.scout_and_set_benchmark()
        self.reset_game()

    def reset_game(self, to_default=False):
        if to_default:
            self.env.reset_to_default_layout()
            self.rl_scout_result = self.manager.scout_and_set_benchmark()
        else:
            self.env.restart_current_level()
        self.current_trajectory = []
        self.last_action = None
        return self.get_state_dict()

    def restart_level_after_failure(self):
        """Restarts the exact same level from beginning after human fails."""
        self.env.restart_current_level()
        self.current_trajectory = []
        self.last_action = None
        return self.get_state_dict()

    def randomize_scenario(self, randomize_maze=True, randomize_agents=True, randomize_fires=True, num_fires=2, fire_spread=False):
        self.env.fire_spread_prob = 0.12 if fire_spread else 0.0
        self.env.randomize_scenario(
            randomize_maze=randomize_maze,
            randomize_agents=randomize_agents,
            randomize_fires=randomize_fires,
            num_fire_clusters=num_fires
        )
        self.rl_scout_result = self.manager.scout_and_set_benchmark()
        self.current_trajectory = []
        self.last_action = None
        return self.get_state_dict()

    def step(self, action, is_human=True, human_feedback=0.0):
        curr_state = self.env.state
        has_ranger_before = self.env.has_ranger
        s_idx = self.env._state_to_idx(curr_state, has_ranger_before)

        obs, reward, term, trunc, info = self.env.step(action, human_feedback=human_feedback)
        next_state = self.env.state
        has_ranger_after = self.env.has_ranger
        next_s_idx = self.env._state_to_idx(next_state, has_ranger_after)
        done = term or trunc

        step_record = {
            "s_idx": int(s_idx),
            "action": int(action),
            "reward": float(reward),
            "next_s_idx": int(next_s_idx),
            "done": bool(done),
            "is_human": bool(is_human)
        }
        self.current_trajectory.append(step_record)
        self.last_action = action

        if is_human:
            self.agent.learn_from_intervention(self.env, s_idx, action, reward, next_s_idx, done)
        else:
            self.agent.update(self.env, s_idx, action, reward, next_s_idx, done)

        level_cleared = bool(term and info.get("reached_evac", False))
        level_failed = bool(trunc and not term)
        auto_randomized = False

        cleared_benchmark_steps = None
        cleared_max_allowed = None

        if level_cleared:
            cleared_benchmark_steps = int(self.env.level_benchmark_steps)
            cleared_max_allowed = int(self.env.level_benchmark_steps + self.env.step_allowance)
            self.agent.on_maze_cleared()
            if self.auto_randomize_on_success:
                num_fires = min(4, 2 + self.env.successful_runs_count // 2)
                self.env.randomize_scenario(
                    randomize_maze=True,
                    randomize_agents=True,
                    randomize_fires=True,
                    num_fire_clusters=num_fires
                )
                self.rl_scout_result = self.manager.scout_and_set_benchmark()
                self.current_trajectory = []
                self.last_action = None
                auto_randomized = True
        elif level_failed:
            self.env.restart_current_level()
            self.current_trajectory = []
            self.last_action = None

        state_dict = self.get_state_dict(info=info, reward=reward, done=done)
        state_dict["level_cleared"] = level_cleared
        state_dict["level_failed"] = level_failed
        state_dict["auto_randomized"] = auto_randomized
        if level_cleared:
            state_dict["done"] = False
            state_dict["level_benchmark_steps"] = cleared_benchmark_steps
            state_dict["max_allowed_steps"] = cleared_max_allowed
        elif level_failed:
            state_dict["done"] = False
            state_dict["level_benchmark_steps"] = None
            state_dict["max_allowed_steps"] = None
        else:
            state_dict["level_benchmark_steps"] = None
            state_dict["max_allowed_steps"] = None
        return state_dict

    def apply_human_feedback(self, valence):
        if self.last_action is None:
            return {"status": "no_previous_action"}
        s_idx = self.env._state_to_idx(self.env.state, self.env.has_ranger)
        new_q = self.agent.apply_human_feedback(self.env, s_idx, self.last_action, valence)
        return {
            "status": "feedback_applied",
            "valence": valence,
            "new_q_val": float(new_q)
        }

    def learn_from_current_trajectory(self):
        if not self.current_trajectory:
            return {"status": "no_trajectory", "updates": 0}
        updates = self.agent.learn_from_demonstration(self.env, self.current_trajectory, epochs=10)
        self.human_demonstrations.append(len(self.current_trajectory))
        return {
            "status": "success",
            "updates": updates,
            "trajectory_length": len(self.current_trajectory),
            "total_demonstrations": len(self.human_demonstrations)
        }

    def train_quick(self, episodes=50):
        returns = self.agent.quick_fine_tune(self.env, episodes=episodes)
        return {
            "status": "success",
            "episodes": episodes,
            "mean_return": float(np.mean(returns[-10:])),
            "final_epsilon": float(self.agent.epsilon)
        }

    def extinguish(self):
        success, rew = self.env.extinguish_current_cell()
        return {
            "success": success,
            "reward": rew,
            "water_remaining": self.env.water_remaining,
            "fires_extinguished": self.env.fires_extinguished
        }

    def get_state_dict(self, info=None, reward=0.0, done=False):
        policy_grid, val_grid = self.agent.get_policy_and_values(self.env, has_ranger=self.env.has_ranger)
        valid_vals = [float(v) for r in val_grid for v in r if v > -900]
        v_min = float(min(valid_vals)) if valid_vals else 0.0
        v_max = float(max(valid_vals)) if valid_vals else 1.0

        safe_info = {}
        for k, v in (info or {}).items():
            if k == "relational_obs":
                continue
            if isinstance(v, (int, np.integer)):
                safe_info[k] = int(v)
            elif isinstance(v, (float, np.floating)):
                safe_info[k] = float(v)
            elif isinstance(v, (bool, np.bool_)):
                safe_info[k] = bool(v)
            elif isinstance(v, (list, tuple)):
                safe_info[k] = [int(x) if isinstance(x, (int, np.integer)) else float(x) if isinstance(x, (float, np.floating)) else str(x) for x in v]
            else:
                safe_info[k] = str(v)

        return {
            "drone_pos": [int(self.env.state[0]), int(self.env.state[1])],
            "start_pos": [int(self.env.start_pos[0]), int(self.env.start_pos[1])],
            "ranger_pos": [int(self.env.ranger_pos[0]), int(self.env.ranger_pos[1])],
            "evac_pos": [int(self.env.evac_pos[0]), int(self.env.evac_pos[1])],
            "has_ranger": bool(self.env.has_ranger),
            "current_step": int(self.env.current_step),
            "level_benchmark_steps": None,
            "step_allowance": int(self.env.step_allowance),
            "max_allowed_steps": None,
            "water_remaining": int(self.env.water_remaining),
            "fires_extinguished": int(self.env.fires_extinguished),
            "current_level": int(self.env.current_level),
            "successful_runs": int(self.env.successful_runs_count),
            "thermal_map": self.env.thermal_map.tolist(),
            "obstacles": [[int(r), int(c)] for r, c in self.env.obstacles],
            "policy_grid": policy_grid.tolist(),
            "value_grid": val_grid.tolist(),
            "v_range": [v_min, v_max],
            "reward": float(reward),
            "done": bool(done),
            "info": safe_info,
            "agent_stats": {
                "interventions": int(self.agent.human_interventions_count),
                "feedbacks": int(self.agent.human_feedbacks_count),
                "demonstrations": int(self.agent.demonstrations_learned_count),
                "mazes_cleared": int(self.agent.mazes_cleared_count),
                "epsilon": float(self.agent.epsilon)
            }
        }


ENGINE = GameEngine()


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Autonomous Anti-Fire Drone Rescue | Blind AI Step Challenge</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card: #131b2e;
      --card-border: #1e293b;
      --accent: #ff4757;
      --drone: #00d2d3;
      --ranger: #ffa502;
      --evac: #2ed573;
      --text: #f1f2f6;
      --text-muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; padding: 18px 24px; display: flex; flex-direction: column; }
    header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 16px; border-bottom: 1px solid var(--card-border); margin-bottom: 18px; }
    .title-group h1 { font-size: 24px; font-weight: 800; background: linear-gradient(135deg, #ff6b6b, #ffa502, #00d2d3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .title-group p { font-size: 13px; color: var(--text-muted); margin-top: 3px; }
    .badge { background: #1e293b; color: #38bdf8; font-size: 11px; font-weight: 700; padding: 4px 10px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.5px; border: 1px solid #334155; }

    .main-grid { display: grid; grid-template-columns: 560px 1fr; gap: 20px; align-items: start; }
    .panel { background: var(--card); border: 1px solid var(--card-border); border-radius: 14px; padding: 18px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }

    .canvas-container { position: relative; width: 520px; height: 520px; margin: 0 auto; background: #080d16; border-radius: 12px; overflow: hidden; border: 2px solid #1e293b; }
    canvas { width: 100%; height: 100%; image-rendering: pixelated; display: block; }

    /* Floating Banners */
    .banner-overlay {
      position: absolute; top: 20px; left: 50%; transform: translateX(-50%);
      padding: 12px 24px; border-radius: 999px; text-align: center;
      font-weight: 800; font-size: 14px; z-index: 25; display: none;
      box-shadow: 0 10px 30px rgba(0,0,0,0.6); pointer-events: none;
    }
    .banner-success { background: linear-gradient(135deg, #10b981, #059669); color: white; border: 2px solid #34d399; }
    .banner-fail { background: linear-gradient(135deg, #ef4444, #b91c1c); color: white; border: 2px solid #f87171; }

    .hud-bar { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 14px; }
    .hud-stat { background: #0b1120; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 10px; text-align: center; }
    .hud-label { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase; }
    .hud-val { font-size: 18px; font-weight: 800; font-family: 'JetBrains Mono', monospace; margin-top: 2px; }

    .btn-group { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    button { background: #1e293b; color: var(--text); border: 1px solid #334155; border-radius: 8px; padding: 8px 14px; font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s ease; display: inline-flex; align-items: center; gap: 6px; }
    button:hover { background: #334155; border-color: #475569; transform: translateY(-1px); }
    button:active { transform: translateY(0); }
    button.primary { background: #2563eb; border-color: #3b82f6; }
    button.primary:hover { background: #1d4ed8; }
    button.danger { background: #dc2626; border-color: #ef4444; }
    button.danger:hover { background: #b91c1c; }
    button.success { background: #16a34a; border-color: #22c55e; }
    button.success:hover { background: #15803d; }
    button.warn { background: #d97706; border-color: #f59e0b; }
    button.warn:hover { background: #b45309; }

    .tabs { display: flex; gap: 6px; border-bottom: 1px solid var(--card-border); padding-bottom: 10px; margin-bottom: 14px; }
    .tab-btn { background: transparent; border: none; border-bottom: 2px solid transparent; color: var(--text-muted); padding: 6px 12px; border-radius: 0; cursor: pointer; font-size: 13px; font-weight: 700; }
    .tab-btn.active { color: #38bdf8; border-bottom-color: #38bdf8; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    .dpad-container { display: grid; grid-template-columns: repeat(3, 48px); grid-template-rows: repeat(3, 48px); gap: 6px; margin: 12px auto; width: 156px; }
    .dpad-btn { width: 48px; height: 48px; padding: 0; font-size: 18px; font-weight: 800; display: flex; align-items: center; justify-content: center; }

    .log-box { background: #070b14; border: 1px solid #1e293b; border-radius: 8px; padding: 10px; font-family: 'JetBrains Mono', monospace; font-size: 11px; height: 160px; overflow-y: auto; color: #a5b4fc; }
    .log-entry { margin-bottom: 4px; line-height: 1.4; }
    .log-entry.alert { color: #f87171; font-weight: 700; }
    .log-entry.success { color: #4ade80; font-weight: 700; }
    .log-entry.info { color: #38bdf8; font-weight: 600; }

    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; font-size: 11px; }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-box { width: 12px; height: 12px; border-radius: 3px; }

    .challenge-card { background: rgba(15, 23, 42, 0.8); border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 12px; }
  </style>
</head>
<body>

  <header>
    <div class="title-group">
      <h1>Autonomous Anti-Fire Drone Rescue System</h1>
      <p>Blind Minimum Steps Challenge • Predict & Fly Optimal Route (+2 Step Allowance)</p>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span class="badge" style="color: #fbbf24; border-color: #fbbf24;" id="level-badge">LEVEL 1 • STREAK: 0</span>
      <span class="badge" style="color: #38bdf8; border-color: #38bdf8;">BLIND AI CHALLENGE (+2 ALLOWANCE)</span>
      <span class="badge" id="mission-phase">PHASE 1: SEARCH</span>
    </div>
  </header>

  <div class="main-grid">
    <!-- Left Panel: Simulation Canvas & Direct HUD -->
    <div class="panel">
      <div class="canvas-container">
        <div id="banner-success" class="banner-overlay banner-success">
          🎉 LEVEL CLEARED!<br><span style="font-size: 12px; font-weight: 500;" id="banner-success-sub">Generating next level...</span>
        </div>
        <div id="banner-fail" class="banner-overlay banner-fail">
          ❌ LEVEL FAILED! EXCEEDED STEP ALLOWANCE!<br><span style="font-size: 12px; font-weight: 500;" id="banner-fail-sub">Restarting this exact level from beginning...</span>
        </div>
        <canvas id="gameCanvas" width="600" height="600"></canvas>
      </div>

      <div class="hud-bar">
        <div class="hud-stat">
          <div class="hud-label">Steps Taken</div>
          <div class="hud-val" id="hud-steps-taken" style="color: #38bdf8;">0</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Step Limit</div>
          <div class="hud-val" style="color: #fbbf24;">HIDDEN (+2)</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Ranger Status</div>
          <div class="hud-val" id="hud-ranger" style="color: var(--ranger);">TRAPPED</div>
        </div>
        <div class="hud-stat">
          <div class="hud-label">Thermal Hazard</div>
          <div class="hud-val" id="hud-hazard" style="color: #4ade80;">0.00</div>
        </div>
      </div>

      <div class="legend">
        <div class="legend-item"><div class="legend-box" style="background: #00d2d3;"></div> Drone (You)</div>
        <div class="legend-item"><div class="legend-box" style="background: #ffa502;"></div> Trapped Ranger</div>
        <div class="legend-item"><div class="legend-box" style="background: #2ed573;"></div> Hospital Base</div>
        <div class="legend-item"><div class="legend-box" style="background: #ff4757;"></div> Wildfire Core</div>
        <div class="legend-item"><div class="legend-box" style="background: #e67e22;"></div> Smoke Plume</div>
        <div class="legend-item"><div class="legend-box" style="background: #2c3e50;"></div> Mountain Cliff</div>
      </div>

      <div class="btn-group" style="justify-content: center; margin-top: 14px;">
        <button id="btn-toggle-policy">🎯 Toggle Policy Arrows</button>
        <button id="btn-drop-water" class="warn">🧯 Suppress Fire (Space)</button>
        <button id="btn-retry-level" class="danger">🔄 Restart Current Level</button>
      </div>
    </div>

    <!-- Right Panel: Challenge Controls, Learning & Logs -->
    <div class="panel">
      <div class="challenge-card">
        <div style="font-size: 12px; font-weight: 800; color: #38bdf8; margin-bottom: 3px;">
          🎯 BLIND MINIMUM STEPS PROTOCOL
        </div>
        <p style="font-size: 11px; color: #cbd5e1; line-height: 1.5;">
          The RL agent scouted this procedural level in secret. <b>No path or step limit is shown!</b>
          You must predict and execute the shortest route yourself (+2 step allowance). If you fail, the AI's benchmark is <b>not revealed</b> and you must retry the exact same maze. You only discover the AI's step count when you win!
        </p>
      </div>

      <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('pilot')">🎮 Human Pilot</button>
        <button class="tab-btn" onclick="switchTab('hitl')">🧠 Human-in-the-Loop RL</button>
      </div>

      <!-- Tab 1: Human Pilot Controls -->
      <div id="tab-pilot" class="tab-content active">
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
          Use <b>Arrow Keys</b> or <b>W/A/S/D</b> to fly. Predict the optimal route!
        </p>

        <div class="dpad-container">
          <div></div>
          <button class="dpad-btn" onclick="sendAction(0)">↑</button>
          <div></div>
          <button class="dpad-btn" onclick="sendAction(2)">←</button>
          <button class="dpad-btn primary" onclick="extinguishFire()">🧯</button>
          <button class="dpad-btn" onclick="sendAction(3)">→</button>
          <div></div>
          <button class="dpad-btn" onclick="sendAction(1)">↓</button>
          <div></div>
        </div>

        <div class="btn-group" style="margin-top: 16px;">
          <button onclick="learnFromFlight()" class="success">🎓 Model: Learn from My Run</button>
          <button onclick="restartLevel()" class="danger">🔄 Restart Level (Same Maze)</button>
          <button onclick="randomizeAll()" class="warn">🎲 Skip to Next Level</button>
        </div>
      </div>

      <!-- Tab 2: Human-in-the-Loop RL -->
      <div id="tab-hitl" class="tab-content">
        <p style="font-size: 13px; color: var(--text-muted); margin-bottom: 12px;">
          Train and guide the AI agent using real-time feedback:
        </p>

        <div style="background: #090e1a; padding: 12px; border-radius: 8px; border: 1px solid #1e293b; margin-bottom: 12px;">
          <div style="font-size: 12px; font-weight: 700; color: #f59e0b; margin-bottom: 4px;">Instant RLHF Feedback</div>
          <div style="display: flex; gap: 6px;">
            <button onclick="sendFeedback(1.0)" class="success" style="flex: 1;">👍 Praise (+1)</button>
            <button onclick="sendFeedback(-1.0)" class="danger" style="flex: 1;">👎 Scold (-1)</button>
          </div>
        </div>

        <div class="btn-group">
          <button onclick="quickTrain(40)" class="warn" style="width: 100%;">⚡ Fast-Tune RL on Current Maze</button>
        </div>
      </div>

      <!-- Live Mission Log -->
      <div style="margin-top: 16px;">
        <div style="font-size: 12px; font-weight: 700; color: var(--text-muted); margin-bottom: 6px; text-transform: uppercase;">Live Mission Stream</div>
        <div class="log-box" id="log-box">
          <div class="log-entry info">[START] AI scouted level in secret (+2 allowance granted). Predict the minimum path!</div>
        </div>
      </div>
    </div>
  </div>

  <script>
    const canvas = document.getElementById('gameCanvas');
    const ctx = canvas.getContext('2d');
    const SIZE = 15;
    const CELL_SIZE = canvas.width / SIZE;

    let gameState = null;
    let showPolicy = false;
    let isActionPending = false;
    let isLevelTransitioning = false;

    document.getElementById('btn-toggle-policy').addEventListener('click', () => {
      showPolicy = !showPolicy;
      render();
    });

    document.getElementById('btn-drop-water').addEventListener('click', extinguishFire);
    document.getElementById('btn-retry-level').addEventListener('click', restartLevel);

    window.addEventListener('keydown', (e) => {
      if (isActionPending || isLevelTransitioning) return;
      if (['ArrowUp', 'KeyW'].includes(e.code)) { e.preventDefault(); sendAction(0); }
      else if (['ArrowDown', 'KeyS'].includes(e.code)) { e.preventDefault(); sendAction(1); }
      else if (['ArrowLeft', 'KeyA'].includes(e.code)) { e.preventDefault(); sendAction(2); }
      else if (['ArrowRight', 'KeyD'].includes(e.code)) { e.preventDefault(); sendAction(3); }
      else if (e.code === 'Space') { e.preventDefault(); extinguishFire(); }
    });

    function log(msg, type='') {
      const box = document.getElementById('log-box');
      const div = document.createElement('div');
      div.className = 'log-entry ' + type;
      div.innerText = `[${new Date().toLocaleTimeString()}] ${msg}`;
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      event.target.classList.add('active');
      document.getElementById(`tab-${name}`).classList.add('active');
    }

    async function fetchState() {
      try {
        const res = await fetch('/api/state');
        gameState = await res.json();
        updateHUD();
        render();
      } catch (err) {
        console.error("Fetch error", err);
      }
    }

    function updateHUD() {
      if (!gameState) return;
      document.getElementById('hud-steps-taken').innerText = gameState.current_step;
      document.getElementById('level-badge').innerText = `LEVEL ${gameState.current_level} • STREAK: ${gameState.successful_runs}`;

      const rBadge = document.getElementById('hud-ranger');
      if (gameState.has_ranger) {
        rBadge.innerText = 'SECURED';
        rBadge.style.color = '#2ed573';
        document.getElementById('mission-phase').innerText = 'PHASE 2: EVACUATION';
      } else {
        rBadge.innerText = 'TRAPPED';
        rBadge.style.color = '#ffa502';
        document.getElementById('mission-phase').innerText = 'PHASE 1: SEARCH';
      }
      document.getElementById('hud-hazard').innerText = (gameState.info.thermal_damage || 0).toFixed(2);
    }

    function drawSpeechBubble(text, targetX, targetY, isUrgent = false) {
      ctx.save();
      ctx.font = 'bold 10px Outfit, sans-serif';
      const textWidth = ctx.measureText(text).width;
      const bubbleW = textWidth + 14;
      const bubbleH = 20;
      let bx = targetX - bubbleW / 2;
      let by = targetY - CELL_SIZE * 0.72 - bubbleH;
      bx = Math.max(6, Math.min(canvas.width - bubbleW - 6, bx));
      by = Math.max(6, by);

      ctx.fillStyle = isUrgent ? 'rgba(220, 38, 38, 0.95)' : 'rgba(15, 23, 42, 0.94)';
      ctx.strokeStyle = isUrgent ? '#fef08a' : '#f59e0b';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (ctx.roundRect) {
        ctx.roundRect(bx, by, bubbleW, bubbleH, 6);
      } else {
        ctx.rect(bx, by, bubbleW, bubbleH);
      }
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(targetX - 4, by + bubbleH);
      ctx.lineTo(targetX, by + bubbleH + 5);
      ctx.lineTo(targetX + 4, by + bubbleH);
      ctx.fillStyle = isUrgent ? 'rgba(220, 38, 38, 0.95)' : 'rgba(15, 23, 42, 0.94)';
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, bx + bubbleW / 2, by + bubbleH / 2 + 1);
      ctx.restore();
    }

    function render(timestamp) {
      if (!gameState) return;
      const t = timestamp || performance.now();
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Grid
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const x = c * CELL_SIZE;
          const y = r * CELL_SIZE;
          ctx.fillStyle = (r + c) % 2 === 0 ? '#101726' : '#0c1322';
          ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
          ctx.strokeStyle = '#1e293b';
          ctx.lineWidth = 0.5;
          ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        }
      }

      // 2. Obstacles (Mountain Cliffs)
      gameState.obstacles.forEach(([r, c]) => {
        const x = c * CELL_SIZE;
        const y = r * CELL_SIZE;
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1.2;
        ctx.strokeRect(x, y, CELL_SIZE, CELL_SIZE);
        ctx.beginPath();
        ctx.moveTo(x, y); ctx.lineTo(x + CELL_SIZE, y + CELL_SIZE);
        ctx.moveTo(x, y + CELL_SIZE); ctx.lineTo(x + CELL_SIZE, y);
        ctx.stroke();
      });

      // 3. Wildfires & Smoke (ANIMATED FLICKER EFFECT)
      for (let r = 0; r < SIZE; r++) {
        for (let c = 0; c < SIZE; c++) {
          const heat = gameState.thermal_map[r][c];
          if (heat > 0) {
            const x = c * CELL_SIZE;
            const y = r * CELL_SIZE;
            if (heat >= 0.8) {
              // Flame cell - square position remains fixed!
              const flicker = 0.80 + 0.18 * Math.sin(t * 0.009 + r * 3.4 + c * 2.1);
              const grad = ctx.createRadialGradient(
                x + CELL_SIZE * 0.5, y + CELL_SIZE * 0.65, CELL_SIZE * 0.1,
                x + CELL_SIZE * 0.5, y + CELL_SIZE * 0.5, CELL_SIZE * 0.72
              );
              grad.addColorStop(0, `rgba(254, 240, 138, ${Math.min(1, flicker)})`);
              grad.addColorStop(0.35, `rgba(249, 115, 22, ${flicker * 0.95})`);
              grad.addColorStop(1, `rgba(220, 38, 38, ${flicker * 0.85})`);
              ctx.fillStyle = grad;
              ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);

              // Floating ember sparks within cell
              const emberPhase = (t * 0.028 + (r * 17 + c * 31)) % CELL_SIZE;
              const emberX = x + CELL_SIZE * 0.5 + Math.sin(t * 0.006 + r + c) * (CELL_SIZE * 0.28);
              const emberY = y + CELL_SIZE - emberPhase;
              const emberAlpha = Math.max(0, 1.0 - (emberPhase / CELL_SIZE));
              ctx.fillStyle = `rgba(254, 240, 138, ${emberAlpha * 0.85})`;
              ctx.fillRect(emberX, emberY, 2, 2);

              // Flickering & pulsing flame emoji
              const flameScale = 1.0 + 0.15 * Math.sin(t * 0.011 + r * 2.3 + c * 4.7);
              const flameY = y + CELL_SIZE * 0.5 + Math.sin(t * 0.014 + c * 3) * 1.4;
              ctx.save();
              ctx.translate(x + CELL_SIZE * 0.5, flameY);
              ctx.scale(flameScale, flameScale);
              ctx.font = '16px Outfit, sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('🔥', 0, 0);
              ctx.restore();
            } else {
              // Smoke cell - gentle drifting opacity
              const smokeAlpha = 0.38 + 0.16 * Math.sin(t * 0.005 + r * 1.8 + c * 1.5);
              ctx.fillStyle = `rgba(243, 156, 18, ${smokeAlpha})`;
              ctx.fillRect(x, y, CELL_SIZE, CELL_SIZE);
              const smokeY = y + CELL_SIZE * 0.5 + Math.sin(t * 0.006 + r + c) * 1.3;
              ctx.font = '13px Outfit, sans-serif';
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillText('💨', x + CELL_SIZE * 0.5, smokeY);
            }
          }
        }
      }

      // 4. Policy Arrows (if toggled)
      if (showPolicy) {
        ctx.font = '14px Outfit, sans-serif';
        ctx.fillStyle = '#38bdf8';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        for (let r = 0; r < SIZE; r++) {
          for (let c = 0; c < SIZE; c++) {
            const arrow = gameState.policy_grid[r][c];
            if (arrow && arrow !== '■') {
              ctx.fillText(arrow, c * CELL_SIZE + CELL_SIZE / 2, r * CELL_SIZE + CELL_SIZE / 2);
            }
          }
        }
      }

      // 5. Start Base
      const [sr, sc] = gameState.start_pos;
      ctx.fillStyle = '#2563eb';
      ctx.beginPath();
      ctx.arc(sc * CELL_SIZE + CELL_SIZE / 2, sr * CELL_SIZE + CELL_SIZE / 2, CELL_SIZE * 0.38, 0, Math.PI * 2);
      ctx.fill();

      // 6. Hospital Base
      const [er, ec] = gameState.evac_pos;
      ctx.fillStyle = '#16a34a';
      ctx.fillRect(ec * CELL_SIZE + 4, er * CELL_SIZE + 4, CELL_SIZE - 8, CELL_SIZE - 8);
      ctx.font = '18px Outfit, sans-serif';
      ctx.fillStyle = 'white';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('🏥', ec * CELL_SIZE + CELL_SIZE / 2, er * CELL_SIZE + CELL_SIZE / 2);

      // 7. Trapped Ranger (ANIMATED VIBRATION SHIVER & EMERGENCY BEACON)
      if (!gameState.has_ranger) {
        const [rr, rc] = gameState.ranger_pos;
        const rcx = rc * CELL_SIZE + CELL_SIZE / 2;
        const rcy = rr * CELL_SIZE + CELL_SIZE / 2;

        // Expanding Emergency Radar Beacon Ring
        const beaconProgress = (t * 0.0015) % 1;
        const ringRadius = CELL_SIZE * 0.35 + beaconProgress * (CELL_SIZE * 0.75);
        const ringAlpha = (1 - beaconProgress) * 0.85;
        ctx.save();
        ctx.strokeStyle = `rgba(249, 115, 22, ${ringAlpha})`;
        ctx.lineWidth = 2;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.arc(rcx, rcy, ringRadius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        // Vibration / Shivering Effect
        const shiverAmp = 1.8;
        const shiverX = Math.sin(t * 0.05) * shiverAmp + Math.cos(t * 0.1) * 0.7;
        const shiverY = Math.cos(t * 0.055) * shiverAmp * 0.7;

        ctx.fillStyle = '#ea580c';
        ctx.beginPath();
        ctx.arc(rcx + shiverX, rcy + shiverY, CELL_SIZE * 0.36, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#fef08a';
        ctx.lineWidth = 1.5;
        ctx.stroke();

        ctx.font = '16px Outfit, sans-serif';
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('🆘', rcx + shiverX, rcy + shiverY);

        // Ranger Distress Messages / Speech Bubble
        const dist = Math.abs(gameState.drone_pos[0] - rr) + Math.abs(gameState.drone_pos[1] - rc);
        let msg = "MAYDAY! OVER HERE!";
        let isUrgent = true;

        if (dist <= 1) {
          msg = "PREPARE HOIST! LAND!";
          isUrgent = false;
        } else if (dist <= 3) {
          msg = "I HEAR YOUR ROTORS!";
          isUrgent = false;
        } else {
          const msgs = [
            "MAYDAY! OVER HERE!",
            "SMOKE RISING FAST!",
            "CANYON ENCLOSED! SOS!",
            "BATTERY WEAKENING!",
            "CAN YOU HEAR ME?!"
          ];
          const idx = Math.floor(t / 2200) % msgs.length;
          msg = msgs[idx];
        }
        drawSpeechBubble(msg, rcx, rcy, isUrgent);
      }

      // 8. Drone (Human Pilot)
      const [dr, dc] = gameState.drone_pos;
      const hoverY = Math.sin(t * 0.008) * 1.5;
      const dx = dc * CELL_SIZE + CELL_SIZE / 2;
      const dy = dr * CELL_SIZE + CELL_SIZE / 2 + hoverY;

      // Rescue aura / green shield if ranger is aboard
      if (gameState.has_ranger) {
        const shieldPulse = (t * 0.002) % 1;
        ctx.save();
        ctx.strokeStyle = `rgba(46, 213, 115, ${0.8 * (1 - shieldPulse)})`;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(dx, dy, CELL_SIZE * 0.42 + shieldPulse * 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
      }

      ctx.fillStyle = gameState.has_ranger ? '#2ed573' : '#00d2d3';
      ctx.beginPath();
      ctx.arc(dx, dy, CELL_SIZE * 0.34, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = 'white';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.font = '14px Outfit, sans-serif';
      ctx.fillStyle = 'black';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(gameState.has_ranger ? '🚁' : '🛸', dx, dy);

      // Speech bubble when ranger rescued aboard drone
      if (gameState.has_ranger) {
        drawSpeechBubble("SECURED! TO HOSPITAL 🏥", dx, dy, false);
      }
    }

    async function sendAction(act) {
      if (isActionPending || isLevelTransitioning) return;
      isActionPending = true;
      try {
        const prevHadRanger = gameState ? gameState.has_ranger : false;
        const prevLevel = gameState ? gameState.current_level : 1;
        const stepsUsed = (gameState ? gameState.current_step : 0) + 1;
        const res = await fetch('/api/step', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: act, is_human: true })
        });
        gameState = await res.json();
        updateHUD();

        if (!prevHadRanger && gameState.has_ranger) {
          log(`🆘 [RANGER RESCUED] "I am securely aboard! Evacuate immediately to Hospital Helipad 🏥!"`, 'success');
        } else if (!gameState.has_ranger && gameState.drone_pos && gameState.ranger_pos) {
          const d = Math.abs(gameState.drone_pos[0] - gameState.ranger_pos[0]) + Math.abs(gameState.drone_pos[1] - gameState.ranger_pos[1]);
          if (d <= 3 && Math.random() < 0.28) {
            log(`📻 [RANGER RADIO] "I hear your rotors beating through the smoke! Keep coming!"`, 'info');
          }
        }

        if (gameState.level_cleared) {
          isLevelTransitioning = true;
          const banner = document.getElementById('banner-success');
          document.getElementById('banner-success-sub').innerText = 
            `You took ${stepsUsed} steps! AI Benchmark revealed: ${gameState.level_benchmark_steps} steps (Target limit was ${gameState.max_allowed_steps}). Advancing to Level ${gameState.current_level}...`;
          banner.style.display = 'block';
          setTimeout(() => {
            banner.style.display = 'none';
            isLevelTransitioning = false;
            updateHUD();
          }, 1800);
          log(`🎉 LEVEL ${prevLevel} CLEARED! You took ${stepsUsed} steps! Secret AI benchmark revealed: ${gameState.level_benchmark_steps} steps (Target limit was ${gameState.max_allowed_steps}). Level ${gameState.current_level} generated!`, 'success');
        } else if (gameState.level_failed) {
          isLevelTransitioning = true;
          const banner = document.getElementById('banner-fail');
          document.getElementById('banner-fail-sub').innerText = 
            `Step allowance exceeded! The AI benchmark remains secret — figure out the optimal path on your own. Restarting Level ${gameState.current_level}...`;
          banner.style.display = 'block';
          setTimeout(() => {
            banner.style.display = 'none';
            isLevelTransitioning = false;
            updateHUD();
          }, 1800);
          log(`❌ LEVEL ${gameState.current_level} FAILED! You took too many steps. The AI's benchmark is kept secret until you win. Figure out the optimal route! Restarting this level from the beginning...`, 'alert');
        }
      } catch (err) {
        console.error("sendAction error:", err);
      } finally {
        isActionPending = false;
      }
    }

    async function restartLevel() {
      try {
        const res = await fetch('/api/restart_level', { method: 'POST' });
        gameState = await res.json();
        log(`🔄 Level ${gameState.current_level} restarted from beginning.`, 'info');
        updateHUD();
        render();
      } catch (err) {
        console.error(err);
      }
    }

    async function extinguishFire() {
      try {
        const res = await fetch('/api/extinguish', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
          log(`💧 Water deployed! Extinguished flame (+${data.reward.toFixed(1)})`, 'success');
          log(`📻 [RANGER RADIO] "Direct hit with water drop! Thermal intensity decreased!"`, 'info');
        }
        fetchState();
      } catch (err) {
        console.error(err);
      }
    }

    async function sendFeedback(valence) {
      try {
        const res = await fetch('/api/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ valence })
        });
        log(`🧠 Feedback ${valence > 0 ? '+1 Praise' : '-1 Penalty'} recorded.`, valence > 0 ? 'success' : 'alert');
        fetchState();
      } catch (err) {
        console.error(err);
      }
    }

    async function learnFromFlight() {
      try {
        const res = await fetch('/api/learn_demonstration', { method: 'POST' });
        const data = await res.json();
        log(`🎓 Q-Learning updated ${data.updates} TD steps from your demonstration!`, 'success');
        showPolicy = true;
        fetchState();
      } catch (err) {
        console.error(err);
      }
    }

    async function quickTrain(episodes) {
      log(`⚡ Fine-tuning RL agent...`);
      try {
        await fetch('/api/train_quick', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ episodes })
        });
        log(`✅ Fine-tuning complete.`, 'success');
        fetchState();
      } catch (err) {
        console.error(err);
      }
    }

    async function randomizeAll() {
      log(`🎲 Generating new procedural level... AI scouting in secret (+2 allowance).`);
      try {
        const res = await fetch('/api/randomize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ randomize_maze: true, randomize_agents: true, randomize_fires: true, num_fires: 2 })
        });
        gameState = await res.json();
        log(`🤖 AI scouted Level ${gameState.current_level} in secret. Plan your minimum steps!`, 'info');
        updateHUD();
        render();
      } catch (err) {
        console.error(err);
      }
    }

    fetchState();

    function animLoop(timestamp) {
      render(timestamp);
      requestAnimationFrame(animLoop);
    }
    requestAnimationFrame(animLoop);
  </script>
</body>
</html>
"""


def _numpy_json_serializer(obj):
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "item"):
        return obj.item()
    return str(obj)


class RequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        try:
            payload = json.dumps(data, default=_numpy_json_serializer).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as err:
            err_payload = json.dumps({"error": str(err)}).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(err_payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(err_payload)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            payload = HTML_TEMPLATE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.path == "/api/state":
            self._send_json(ENGINE.get_state_dict())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}

            if self.path == "/api/step":
                action = body.get("action")
                is_human = body.get("is_human", True)
                human_feedback = float(body.get("human_feedback", 0.0))

                if action is None:
                    s_idx = ENGINE.env._state_to_idx(ENGINE.env.state, ENGINE.env.has_ranger)
                    action = ENGINE.agent.select_action(ENGINE.env, s_idx, greedy=True)
                    is_human = False
                elif isinstance(action, str):
                    act_map = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
                    action = act_map.get(action.strip().upper(), int(action) if action.strip().isdigit() else 0)
                else:
                    action = int(action)

                state = ENGINE.step(action, is_human=is_human, human_feedback=human_feedback)
                self._send_json(state)

            elif self.path == "/api/restart_level":
                state = ENGINE.restart_level_after_failure()
                self._send_json(state)

            elif self.path == "/api/feedback":
                valence = float(body.get("valence", 0.0))
                res = ENGINE.apply_human_feedback(valence)
                self._send_json(res)

            elif self.path == "/api/learn_demonstration":
                res = ENGINE.learn_from_current_trajectory()
                self._send_json(res)

            elif self.path == "/api/train_quick":
                episodes = int(body.get("episodes", 50))
                res = ENGINE.train_quick(episodes=episodes)
                self._send_json(res)

            elif self.path == "/api/extinguish":
                res = ENGINE.extinguish()
                self._send_json(res)

            elif self.path == "/api/randomize":
                maze = body.get("randomize_maze", True)
                agents = body.get("randomize_agents", True)
                fires = body.get("randomize_fires", True)
                num_fires = int(body.get("num_fires", 2))
                spread = bool(body.get("fire_spread", False))
                state = ENGINE.randomize_scenario(
                    randomize_maze=maze,
                    randomize_agents=agents,
                    randomize_fires=fires,
                    num_fire_clusters=num_fires,
                    fire_spread=spread
                )
                self._send_json(state)

            elif self.path == "/api/reset":
                to_default = body.get("to_default", False)
                state = ENGINE.reset_game(to_default=to_default)
                self._send_json(state)

            else:
                self.send_response(404)
                self.end_headers()
        except Exception as err:
            self._send_json({"error": str(err)}, status=500)


def run_cli_game():
    print("=" * 65)
    print("ANTI-FIRE DRONE RESCUE: TERMINAL CLI EDITION")
    print("Rules: AI benchmark is SECRET until victory (+2 Step Allowance).")
    print("Controls: W=Up | S=Down | A=Left | D=Right | F=Water | R=Retry | Q=Quit")
    print("=" * 65)

    engine = ENGINE

    def render_cli(state_dict, message=""):
        lvl = state_dict.get("current_level", 1)
        steps = state_dict.get("current_step", 0)
        has_ranger = state_dict.get("has_ranger", False)
        water = state_dict.get("water_remaining", 3)
        streak = state_dict.get("successful_runs", 0)

        phase = "PHASE 2: EVACUATE TO HOSPITAL [H]" if has_ranger else "PHASE 1: SEARCH & RESCUE [R]"
        print("\n" + "=" * 55)
        print(f"LEVEL {lvl} | STREAK: {streak} | STEPS TAKEN: {steps} | WATER: {water}/3")
        print(f"MISSION: {phase}")
        print("STEP LIMIT: [SECRET - Predict optimal path (+2 allowance)]")
        print("=" * 55)

        drone = tuple(state_dict["drone_pos"])
        ranger = tuple(state_dict["ranger_pos"])
        evac = tuple(state_dict["evac_pos"])
        start = tuple(state_dict["start_pos"])
        obstacles = set((r, c) for r, c in state_dict["obstacles"])
        thermal = state_dict["thermal_map"]

        d_dist = abs(drone[0] - ranger[0]) + abs(drone[1] - ranger[1])
        if not has_ranger:
            if d_dist <= 2:
                print(">> [RANGER RADIO] 'I see your searchlights! Land right here!'")
            elif d_dist <= 4:
                print(">> [RANGER RADIO] 'I hear your rotors cutting through the smoke!'")
            else:
                ranger_distress = [
                    ">> [RANGER (SHIVERING)] 'MAYDAY! Canyon pocket enclosed by flames!'",
                    ">> [RANGER (BEACON PULSING)] 'Smoke rising rapidly! Thermal corridor narrow!'",
                    ">> [RANGER (TRANSCEIVER)] 'Battery fading, please hurry!'"
                ]
                print(ranger_distress[steps % len(ranger_distress)])
        else:
            print(">> [RANGER RESCUED] 'Aboard the drone! Transport to Hospital Helipad!'")

        if message:
            print(f">> {message}")

        print("-" * 55)
        print("     " + " ".join(f"{c % 10}" for c in range(15)))
        for r in range(15):
            row_items = []
            for c in range(15):
                pos = (r, c)
                if pos == drone:
                    row_items.append("D+" if has_ranger else "D ")
                elif not has_ranger and pos == ranger:
                    row_items.append("R ")
                elif pos == evac:
                    row_items.append("H ")
                elif pos == start:
                    row_items.append("S ")
                elif pos in obstacles:
                    row_items.append("# ")
                elif thermal[r][c] >= 0.8:
                    row_items.append("* ")
                elif thermal[r][c] > 0.0:
                    row_items.append("~ ")
                else:
                    row_items.append(". ")
            print(f"{r:2d} | " + " ".join(row_items))
        print("-" * 55)
        print("Legend: D=Drone, R=Ranger, H=Hospital, S=Start, #=Wall, *=Fire, ~=Smoke")

    msg = ""
    while True:
        state = engine.get_state_dict()
        render_cli(state, message=msg)
        msg = ""

        try:
            cmd = input("\nEnter Action [W/A/S/D=Move, F=Water, R=Retry, Q=Quit]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI game.")
            break

        if cmd == 'q':
            print("Thanks for playing!")
            break
        elif cmd == 'r':
            state = engine.restart_level_after_failure()
            msg = "Level restarted from beginning."
            continue
        elif cmd == 'f':
            res = engine.extinguish()
            if res.get("success"):
                msg = f"Water cannon discharged! Extinguished flame (+{res.get('reward', 0.0):.1f})"
            else:
                msg = "No fire here or water cannon empty."
            continue

        act_map = {'w': 0, 's': 1, 'a': 2, 'd': 3, 'up': 0, 'down': 1, 'left': 2, 'right': 3}
        if cmd not in act_map:
            msg = f"Unknown command '{cmd}'. Use W, A, S, D, F, R, or Q."
            continue

        steps_before = state.get("current_step", 0) + 1
        prev_lvl = state.get("current_level", 1)
        state = engine.step(act_map[cmd], is_human=True)

        if state.get("level_cleared"):
            print("\n" + "*" * 65)
            print(f"LEVEL {prev_lvl} CLEARED IN {steps_before} STEPS!")
            print(f"Secret AI Benchmark Revealed: {state.get('level_benchmark_steps')} steps")
            print(f"Target Allowance Was: {state.get('max_allowed_steps')} steps")
            print(f"Advancing to Level {state.get('current_level')}!")
            print("*" * 65)
            try:
                input("Press [Enter] to continue to next level...")
            except (KeyboardInterrupt, EOFError):
                break
        elif state.get("level_failed"):
            print("\n" + "!" * 65)
            print("LEVEL FAILED! You took too many steps.")
            print("The AI benchmark remains SECRET so you can figure it out on your own.")
            print(f"Restarting Level {state.get('current_level')} from the beginning...")
            print("!" * 65)
            try:
                input("Press [Enter] to retry this level...")
            except (KeyboardInterrupt, EOFError):
                break


def start_server():
    server = HTTPServer((HOST, PORT), RequestHandler)
    url = f"http://{HOST}:{PORT}"
    print("=" * 65)
    print("ANTI-FIRE DRONE RESCUE: BLIND AI BENCHMARK & +2 ALLOWANCE SERVER")
    print(f"Access Dashboard & Game at: {url}")
    print("=" * 65)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down interactive server.")
        server.server_close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ("--cli", "-c", "cli"):
        run_cli_game()
    else:
        start_server()
