"""
generate_demo_gif.py
Generates:
1. Individual well-labeled GIFs for EACH model:
   - model_1_bfs.gif
   - model_2_astar_manhattan.gif
   - model_3_astar_risk_aware.gif
   - model_4_q_learning.gif
   - model_5_ppo.gif
   - model_6_sac.gif
2. Comparative Search vs RL Side-by-Side GIF:
   - drone_comparison_search_vs_rl.gif
3. Master Synchronized Multi-Model Comparison GIF (4-Quadrant Grid):
   - master_comparison_all_models.gif
   - drone_rescue_demo.gif (submission alias)
"""

import os
import io
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from wildfire_drone_env import WildfireDroneRescueEnv
from run_experiments import (
    run_bfs, run_astar, train_q_learning, train_ppo, train_sac,
    extract_rl_path, SEED
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def _draw_base_map(ax, env, step_i=0, pickup_step=None):
    """Draws the 15x15 terrain background, fires, obstacles, and landmarks."""
    cmap_heat = sns.color_palette("YlOrRd", as_cmap=True)
    ax.imshow(env.thermal_map, cmap=cmap_heat, vmin=0, vmax=1.0, extent=[0, env.size, env.size, 0], alpha=0.92)

    # Obstacles (Mountain Cliffs)
    for r, c in env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, facecolor="#1e272c", edgecolor="#37474f", lw=1.0, hatch="//"))

    # Fire Zone Badges
    ax.text(5.5, 3.2, "WILDFIRE ZONE 1\n(Entrance Gorge)", ha="center", va="center",
            color="#b71c1c", fontweight="bold", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", alpha=0.88, edgecolor="#ef9a9a", lw=0.8))
    
    ax.text(8.5, 9.2, "WILDFIRE ZONE 2\n(Egress Blaze)", ha="center", va="center",
            color="#b71c1c", fontweight="bold", fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#ffebee", alpha=0.88, edgecolor="#ef9a9a", lw=0.8))

    # Depot (0, 0)
    ax.plot(0.5, 0.5, marker="h", color="#1565c0", markersize=13, markeredgecolor="white", markeredgewidth=1.6)
    ax.text(0.5, 0.15, "LAUNCH BASE\n(0,0)", ha="center", va="bottom", color="#0d47a1", fontweight="bold", fontsize=7.5)

    # Hospital Evac Base (14, 14)
    ax.plot(14.5, 14.5, marker="*", color="#2e7d32", markersize=15, markeredgecolor="white", markeredgewidth=1.6)
    ax.text(14.5, 14.15, "HOSPITAL EVAC\n(14,14)", ha="center", va="bottom", color="#1b5e20", fontweight="bold", fontsize=7.5)

    # Ranger Pocket (7, 2)
    has_picked_up = (pickup_step is not None and step_i >= pickup_step)
    just_picked_up = (pickup_step is not None and step_i == pickup_step)

    if just_picked_up:
        ax.plot(2.5, 7.5, marker="P", color="#2e7d32", markersize=15, markeredgecolor="gold", markeredgewidth=2.2)
        ax.text(2.5, 7.05, "RANGER SECURED!", ha="center", va="bottom", color="#1b5e20", fontweight="bold", fontsize=8.0,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#e8f8f5", edgecolor="#2ecc71", lw=1.2))
    elif has_picked_up:
        ax.plot(2.5, 7.5, marker="x", color="#78909c", markersize=9, markeredgewidth=2)
        ax.text(2.5, 7.1, "Pocket Cleared", ha="center", va="bottom", color="#546e7a", fontsize=7.0)
    else:
        ax.plot(2.5, 7.5, marker="P", color="#e65100", markersize=13, markeredgecolor="white", markeredgewidth=1.5)
        ax.text(2.5, 7.05, "TRAPPED RANGER\n(7,2)", ha="center", va="bottom", color="#bf360c", fontweight="bold", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", edgecolor="#ffb74d", lw=0.9))


def render_single_agent_gif(env, path, model_name, category_badge, line_color, drone_color, output_filename):
    """Renders a dedicated, highly informative GIF for a single model with live 2-phase telemetry."""
    frames = []

    pickup_step = next((i for i, p in enumerate(path) if tuple(p) == tuple(env.ranger_pos)), None)

    for step_i, curr_pos in enumerate(path):
        fig, ax = plt.subplots(figsize=(7.8, 8.6), dpi=95)

        _draw_base_map(ax, env, step_i, pickup_step)

        has_picked_up = (pickup_step is not None and step_i >= pickup_step)

        # Plot flight trails
        if step_i > 0:
            if pickup_step is not None and step_i > pickup_step:
                p1 = path[:pickup_step+1]
                ax.plot([p[1] + 0.5 for p in p1], [p[0] + 0.5 for p in p1],
                        color=line_color, lw=2.4, alpha=0.75, linestyle="--")
                p2 = path[pickup_step:step_i+1]
                ax.plot([p[1] + 0.5 for p in p2], [p[0] + 0.5 for p in p2],
                        color="#16a085", lw=3.4, alpha=0.95, linestyle="-")
            else:
                p_sub = path[:step_i+1]
                ax.plot([p[1] + 0.5 for p in p_sub], [p[0] + 0.5 for p in p_sub],
                        color=line_color, lw=3.0, alpha=0.88, linestyle="-")

        # Current drone position & telemetry
        r_curr, c_curr = curr_pos[0], curr_pos[1]
        current_heat = float(env.thermal_map[r_curr, c_curr])
        cum_hazard = sum(float(env.thermal_map[p[0], p[1]]) for p in path[:step_i+1])

        # Drone marker
        marker_color = "#c0392b" if current_heat >= 0.8 else ("#e67e22" if current_heat > 0 else (drone_color if not has_picked_up else "#27ae60"))
        ax.plot(c_curr + 0.5, r_curr + 0.5, marker="o", color=marker_color, markersize=16, markeredgecolor="white", markeredgewidth=2.2)

        if has_picked_up:
            ax.plot(c_curr + 0.5, r_curr + 0.5, marker="P", color="white", markersize=8.5)
        else:
            ax.plot(c_curr + 0.5, r_curr + 0.5, marker="^", color="white", markersize=7.5)

        # Status & Phase determination
        is_finished = (step_i == len(path) - 1)
        if is_finished:
            if cum_hazard == 0.0:
                phase_badge = "[MISSION ACCOMPLISHED: ZERO DAMAGE]"
                status_str = "SUCCESS: RANGER SAFELY EVACUATED TO HOSPITAL BASE!"
                status_bg = "#d4efdf"
            else:
                phase_badge = "[MISSION COMPLETED WITH THERMAL DAMAGE]"
                status_str = f"COMPLETED: RANGER EVACUATED BUT INCURRED BURNS ({cum_hazard:.2f})!"
                status_bg = "#fadbd8"
        elif pickup_step is not None and step_i == pickup_step:
            phase_badge = "[RANGER SECURED ONBOARD]"
            status_str = "EXTRACTION COMPLETED: COMMENCING EGRESS TO HOSPITAL BASE"
            status_bg = "#fef9e7"
        elif not has_picked_up:
            phase_badge = "[PHASE 1: SEARCH & INGRESS]"
            if current_heat >= 0.8:
                status_str = "CRITICAL: PENETRATING ACTIVE FLAME CORE!"
                status_bg = "#f9d5d5"
            elif current_heat > 0.0:
                status_str = "CAUTION: NAVIGATING DENSE SMOKE PLUME"
                status_bg = "#fef5e7"
            else:
                status_str = "SAFE INGRESS: PENETRATING CANYON TOWARD RANGER (7,2)"
                status_bg = "#eaf2f8"
        else:
            phase_badge = "[PHASE 2: EVACUATION & EGRESS]"
            if current_heat >= 0.8:
                status_str = "CRITICAL: FLYING RANGER THROUGH ACTIVE FLAMES!"
                status_bg = "#f9d5d5"
            elif current_heat > 0.0:
                status_str = "CAUTION: SMOKE PLUME EXPOSURE DURING EVACUATION"
                status_bg = "#fef5e7"
            else:
                status_str = "SAFE EGRESS: BYPASS CORRIDOR TOWARD BASE (14,14)"
                status_bg = "#e8f8f5"

        # Telemetry HUD Box
        hud_l1 = f"MODEL: {model_name.upper()} ({category_badge})"
        hud_l2 = f"PHASE: {phase_badge} | PAYLOAD: {'Ranger Onboard [MEDEVAC]' if has_picked_up else 'Solo UAV [SEARCH]'}"
        hud_l3 = f"Step: {step_i}/{len(path)-1} | Pos: ({r_curr}, {c_curr}) | Cell Heat: {current_heat:.2f} | Cum. Hazard: {cum_hazard:.2f}"
        hud_l4 = f"STATUS: {status_str}"
        hud_text = f"{hud_l1}\n{hud_l2}\n{hud_l3}\n{hud_l4}"

        ax.text(
            0.5, 17.15, hud_text,
            ha="left", va="bottom", fontsize=8.2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=status_bg, edgecolor="#78909c", lw=1.2)
        )

        ax.set_xlim(0, env.size)
        ax.set_ylim(env.size + 2.5, -0.2)
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.grid(True, color="gray", alpha=0.22, linestyle=":")
        ax.set_title(f"Autonomous Wildfire Drone Mission: {model_name}", fontsize=11.5, fontweight="bold", pad=9)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    durations = [280] * (len(frames) - 1) + [2400]
    out_file = os.path.join(FIGURES_DIR, output_filename)
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  --> Saved: {out_file}")


def render_search_vs_rl_comparison_gif(env, search_model, rl_model):
    """Renders a 2-panel side-by-side comparison GIF: Classical Search vs Reinforcement Learning."""
    print("\nGenerating Classical Search vs RL Side-by-Side Comparison GIF...")

    p_search = search_model["path"]
    p_rl = rl_model["path"]
    max_steps = max(len(p_search), len(p_rl))

    pickup_search = next((i for i, p in enumerate(p_search) if tuple(p) == tuple(env.ranger_pos)), None)
    pickup_rl = next((i for i, p in enumerate(p_rl) if tuple(p) == tuple(env.ranger_pos)), None)

    frames = []

    for step_idx in range(max_steps):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 7.8), dpi=90)

        for ax, p_data, pickup_s, title, is_rl in [
            (ax1, search_model, pickup_search, "CLASSICAL SEARCH: A* (Manhattan)", False),
            (ax2, rl_model, pickup_rl, f"REINFORCEMENT LEARNING: {rl_model['name']}", True),
        ]:
            path = p_data["path"]
            color = p_data["color"]
            curr_i = min(step_idx, len(path) - 1)
            pos_c = path[curr_i]
            r_c, c_c = pos_c[0], pos_c[1]

            _draw_base_map(ax, env, curr_i, pickup_s)

            has_p = (pickup_s is not None and curr_i >= pickup_s)

            if curr_i > 0:
                ax.plot([p[1] + 0.5 for p in path[:curr_i+1]],
                        [p[0] + 0.5 for p in path[:curr_i+1]],
                        color=color, lw=2.8, linestyle="--" if not is_rl else "-")

            heat_c = float(env.thermal_map[r_c, c_c])
            haz_c = sum(float(env.thermal_map[p[0], p[1]]) for p in path[:curr_i+1])
            is_done = (curr_i == len(path) - 1)

            marker_c = "#c0392b" if heat_c >= 0.8 else ("#e67e22" if heat_c > 0 else color)
            ax.plot(c_c + 0.5, r_c + 0.5, marker="o", color=marker_c, markersize=14, markeredgecolor="white", markeredgewidth=2.0)
            if has_p:
                ax.plot(c_c + 0.5, r_c + 0.5, marker="P", color="white", markersize=7.5)
            else:
                ax.plot(c_c + 0.5, r_c + 0.5, marker="^", color="white", markersize=6.5)

            if is_done:
                stat = f"COMPLETED [DAMAGE: {haz_c:.2f}]"
                bg = "#d4efdf" if haz_c == 0 else "#fadbd8"
            elif curr_i == pickup_s:
                stat = "RANGER SECURED ONBOARD"
                bg = "#fef9e7"
            elif not has_p:
                stat = "PHASE 1: SEARCHING RANGER"
                bg = "#f9d5d5" if heat_c > 0 else "#eaf2f8"
            else:
                stat = "PHASE 2: EVACUATION EGRESS"
                bg = "#f9d5d5" if heat_c > 0 else "#e8f8f5"

            hud = f"Step: {curr_i}/{len(path)-1} | Pos: ({r_c},{c_c}) | Damage: {haz_c:.2f}\nStatus: {stat}"
            ax.text(0.5, 16.9, hud, fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", facecolor=bg, edgecolor="#90a4ae", lw=1.0))

            ax.set_xlim(0, env.size); ax.set_ylim(env.size + 2.2, -0.2)
            ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))
            ax.grid(True, color="gray", alpha=0.2, linestyle=":")
            ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)

        plt.suptitle("PARADIGM DIVERGENCE: CLASSICAL OPEN-LOOP SEARCH VS RISK-AWARE RL IN HAZARDOUS TERRAIN",
                     fontsize=12, fontweight="bold", y=0.98)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    durations = [280] * (len(frames) - 1) + [2500]
    out_file = os.path.join(FIGURES_DIR, "drone_comparison_search_vs_rl.gif")
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  --> Saved: {out_file}")


def render_master_comparison_gif(env, models_dict):
    """Renders a 4-Quadrant live synchronized animation comparing key models."""
    print("\nGenerating Master Synchronized Comparison Animation (4-Quadrant Grid)...")

    keys = ["BFS", "A* (Manhattan)", "Q-Learning", "PPO"]
    paths = [models_dict[k]["path"] for k in keys]
    max_steps = max(len(p) for p in paths)

    pickup_steps = {k: next((i for i, p in enumerate(models_dict[k]["path"]) if tuple(p) == tuple(env.ranger_pos)), None)
                    for k in keys}

    frames = []

    for step_idx in range(max_steps):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 14.5), dpi=85)
        axes = axes.flatten()

        for panel_idx, k in enumerate(keys):
            ax = axes[panel_idx]
            info = models_dict[k]
            path = info["path"]
            color = info["color"]
            pickup_s = pickup_steps[k]

            step_curr = min(step_idx, len(path) - 1)
            pos_curr = path[step_curr]
            r_c, c_c = pos_curr[0], pos_curr[1]
            heat_curr = float(env.thermal_map[r_c, c_c])
            cum_haz = sum(float(env.thermal_map[p[0], p[1]]) for p in path[:step_curr+1])
            has_p = (pickup_s is not None and step_curr >= pickup_s)

            _draw_base_map(ax, env, step_curr, pickup_s)

            # Trail
            if step_curr > 0:
                ax.plot([p[1] + 0.5 for p in path[:step_curr+1]],
                        [p[0] + 0.5 for p in path[:step_curr+1]],
                        color=color, lw=2.4, linestyle=info.get("ls", "-"))

            # Drone marker
            marker_c = "#c0392b" if heat_curr >= 0.8 else ("#e67e22" if heat_curr > 0 else color)
            ax.plot(c_c + 0.5, r_c + 0.5, marker="o", color=marker_c, markersize=13, markeredgecolor="white", markeredgewidth=1.8)
            if has_p:
                ax.plot(c_c + 0.5, r_c + 0.5, marker="P", color="white", markersize=7)
            else:
                ax.plot(c_c + 0.5, r_c + 0.5, marker="^", color="white", markersize=6)

            is_done = (step_curr == len(path) - 1)
            if is_done:
                stat = f"RESCUED [DAMAGE: {cum_haz:.2f}]"
                bg = "#d4efdf" if cum_haz == 0 else "#fadbd8"
            elif step_curr == pickup_s:
                stat = "RANGER SECURED ONBOARD"
                bg = "#fef9e7"
            elif not has_p:
                stat = "PHASE 1: SEARCH & INGRESS"
                bg = "#f9d5d5" if heat_curr > 0 else "#eaeded"
            else:
                stat = "PHASE 2: EVACUATION EGRESS"
                bg = "#f9d5d5" if heat_curr > 0 else "#e8f8f5"

            hud = f"Step: {step_curr}/{len(path)-1} | Damage: {cum_haz:.2f} | Status: {stat}"
            ax.text(0.5, 16.9, hud, fontsize=8.0, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.28", facecolor=bg, edgecolor="#90a4ae", lw=0.9))

            ax.set_xlim(0, env.size); ax.set_ylim(env.size + 2.2, -0.2)
            ax.set_xticks(range(env.size)); ax.set_yticks(range(env.size))
            ax.grid(True, color="gray", alpha=0.2, linestyle=":")
            ax.set_title(f"{k} ({info['badge']})", fontsize=10, fontweight="bold", pad=6)

        plt.suptitle("AUTONOMOUS WILDFIRE SEARCH & RESCUE: MULTI-MODEL SYNCHRONIZED RACE",
                     fontsize=12.5, fontweight="bold", y=0.98)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).copy())

    durations = [280] * (len(frames) - 1) + [2500]
    out_file = os.path.join(FIGURES_DIR, "master_comparison_all_models.gif")
    frames[0].save(out_file, save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print(f"  --> Saved: {out_file}")


def generate_all_model_gifs():
    print("=" * 65)
    print("GENERATING INDIVIDUAL AND COMPARATIVE ANIMATIONS FOR ALL MODELS")
    print("=" * 65)

    env = WildfireDroneRescueEnv(reward_mode="risk_shaped", seed=SEED)

    # 1. Classical Search Models
    print("\nRunning Classical Search...")
    res_bfs = run_bfs(env)
    res_astar_man = run_astar(env, heuristic_type="manhattan")
    res_astar_risk = run_astar(env, heuristic_type="risk_aware")

    # 2. Modern RL Models: Q-Learning, PPO, SAC
    print("Training Q-Learning Agent...")
    q_agent, _, _, _, _ = train_q_learning(env, episodes=2000)
    q_path, _, q_haz, _ = extract_rl_path(env, q_agent)

    print("Training PPO Agent...")
    ppo_agent, _, _, _, _ = train_ppo(env, episodes=2000)
    ppo_path, _, ppo_haz, _ = extract_rl_path(env, ppo_agent)

    print("Training SAC Agent...")
    sac_agent, _, _, _, _ = train_sac(env, episodes=2000)
    sac_path, _, sac_haz, _ = extract_rl_path(env, sac_agent)

    models = {
        "BFS": {
            "name": "BFS",
            "path": res_bfs["path"],
            "badge": "Uninformed Classical Search",
            "line_color": "#2980b9",
            "drone_color": "#2980b9",
            "filename": "model_1_bfs.gif",
            "ls": "-",
            "color": "#2980b9"
        },
        "A* (Manhattan)": {
            "name": "A* (Manhattan)",
            "path": res_astar_man["path"],
            "badge": "Heuristic Search (Distance Only)",
            "line_color": "#d35400",
            "drone_color": "#d35400",
            "filename": "model_2_astar_manhattan.gif",
            "ls": "--",
            "color": "#d35400"
        },
        "A* (Risk-Aware)": {
            "name": "A* (Risk-Aware)",
            "path": res_astar_risk["path"],
            "badge": "Heuristic Search (Safety & Detour)",
            "line_color": "#8e44ad",
            "drone_color": "#8e44ad",
            "filename": "model_3_astar_risk_aware.gif",
            "ls": "-.",
            "color": "#8e44ad"
        },
        "Q-Learning": {
            "name": "Q-Learning",
            "path": q_path,
            "badge": "Value-Based Reinforcement Learning",
            "line_color": "#27ae60",
            "drone_color": "#27ae60",
            "filename": "model_4_q_learning.gif",
            "ls": "-",
            "color": "#27ae60"
        },
        "PPO": {
            "name": "PPO",
            "path": ppo_path,
            "badge": "Policy-Gradient / Clipped PPO RL",
            "line_color": "#e7298a",
            "drone_color": "#e7298a",
            "filename": "model_5_ppo.gif",
            "ls": "--",
            "color": "#e7298a"
        },
        "SAC": {
            "name": "SAC",
            "path": sac_path,
            "badge": "Maximum-Entropy Soft Actor-Critic RL",
            "line_color": "#66a61e",
            "drone_color": "#66a61e",
            "filename": "model_6_sac.gif",
            "ls": "-.",
            "color": "#66a61e"
        }
    }

    # Generate individual GIFs for EACH model
    print("\nRendering Individual Model GIFs:")
    for name, data in models.items():
        print(f"  Rendering {name} -> {data['filename']}...")
        render_single_agent_gif(
            env=env,
            path=data["path"],
            model_name=name,
            category_badge=data["badge"],
            line_color=data["line_color"],
            drone_color=data["drone_color"],
            output_filename=data["filename"]
        )

    # Generate Search vs RL side-by-side comparison GIF
    render_search_vs_rl_comparison_gif(env, models["A* (Manhattan)"], models["Q-Learning"])

    # Generate Master Comparison 4-Quadrant Race GIF
    render_master_comparison_gif(env, models)

    # Copy master comparison to drone_rescue_demo.gif as default presentation GIF
    master_comp_gif = os.path.join(FIGURES_DIR, "master_comparison_all_models.gif")
    target_demo_gif = os.path.join(FIGURES_DIR, "drone_rescue_demo.gif")
    if os.path.exists(master_comp_gif):
        shutil.copyfile(master_comp_gif, target_demo_gif)

    print("\n" + "=" * 65)
    print("ALL INDIVIDUAL & COMPARISON ANIMATIONS SUCCESSFULLY GENERATED!")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_model_gifs()
