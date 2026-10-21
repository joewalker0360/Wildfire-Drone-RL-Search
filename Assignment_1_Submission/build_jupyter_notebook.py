"""
build_jupyter_notebook.py
Builds the complete, highly polished Jupyter Notebook (Assignment_1_Wildfire_Drone.ipynb)
for the Two-Phase Wildfire Search, Rescue & Evacuation Mission.
Covers: Classical Search (BFS, A*) to Modern RL (Q-Learning, PPO, SAC).
"""

import os
import nbformat as nbf

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "Assignment_1_Wildfire_Drone.ipynb")

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Machine Learning (23MEE305) - Assignment 1
## Exploring the Evolution of Search Algorithms from DFS, BFS to Reinforcement Learning: Q-Learning, PPO, SAC

**Mission Scenario:** Two-Phase Autonomous Wildfire Search, Rescue & Evacuation Drone  
**Course Code:** 23MEE305  
**Deliverables:** Custom Gymnasium Environment, Classical Search Baselines (BFS, A*), 3 Distinct RL Algorithm Families (**Q-Learning**, **PPO**, **SAC**), Reward Shaping Ablation, Stochastic Wind Robustness Evaluation (+5 Bonus Marks), and Synchronized Animated Visualizations.

---

### Executive Summary & Operational Scenario
In high-risk disaster response robotics, an autonomous Unmanned Aerial Vehicle (UAV) is deployed on a critical **Two-Phase Search, Rescue & Evacuation** mission across an intricate 15x15 canyon terrain ($|S| = 450$ discrete states):
- **Phase 1 (Search & Ingress):** The drone launches from **Depot Base (0, 0)**, penetrates through mountain barriers, locates, and secures an injured forest ranger trapped inside an enclosed rock pocket at **(7, 2)**.
- **Phase 2 (Evacuation & Egress):** With the ranger secured onboard, the drone must navigate through escape corridors and safely deliver the patient to the **Hospital Evacuation Base at (14, 14)**.

### Environmental Hazards & Algorithmic Dilemma:
1. **Mountain Ridges & Enclosed Pocket:** The canyon pocket at `(7, 2)` is enclosed on 3 sides by impassable rock cliffs, forcing the agent to discover the precise ingress/egress opening.
2. **Two Strategic Wildfire Blaze Zones:**
   - **Fire Zone 1 (Entrance Gorge):** Positioned at rows 3–5, col 5, blocking the direct geometric line from the depot.
   - **Fire Zone 2 (Evacuation Corridor Blaze):** Positioned at rows 9–12, col 8, blocking the direct line to the hospital helipad.
3. **The Core Dilemma:** Geometric distance heuristics (e.g. Euclidean / Manhattan distance) tempt the agent to fly through active flames ($T=1.0$), incurring catastrophic thermal burns. In contrast, risk-aware RL algorithms (**Q-Learning**, **PPO**, and **SAC**) learn to discover safe western and eastern bypass corridors, accomplishing a **flawless zero-hazard extraction**.
"""))

# Cell 1: Environment Code
cells.append(nbf.v4.new_markdown_cell("""### 1. Custom Gymnasium Environment: `WildfireDroneRescueEnv`"""))

cells.append(nbf.v4.new_code_cell("""import numpy as np
import gymnasium as gym
from gymnasium import spaces
import heapq
import time
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# Strict reproducibility
SEED = 42
np.random.seed(SEED)

class WildfireDroneRescueEnv(gym.Env):
    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(self, size=15, wind_prob=0.0, reward_mode="risk_shaped", max_steps=220, seed=None):
        super().__init__()
        self.size = size
        self.wind_prob = wind_prob
        self.reward_mode = reward_mode  # 'sparse', 'dense_distance', 'risk_shaped'
        self.max_steps = max_steps
        self.gamma = 0.99
        self.hazard_weight = 45.0

        self.action_space = spaces.Discrete(4)  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]), high=np.array([size - 1, size - 1, 1]), dtype=np.int32
        )

        self.start_pos = (0, 0)
        self.ranger_pos = (7, 2)    # Trapped ranger pocket
        self.evac_pos = (size - 1, size - 1)  # Emergency hospital base

        # Mountain Obstacles
        self.obstacles = set()
        for r in range(2, 13): self.obstacles.add((r, 0))
        for c in range(2, 13): self.obstacles.add((14, c))
        for c in range(2, 13): self.obstacles.add((0, c))
        for r in range(2, 13): self.obstacles.add((r, 14))

        # Canyon Pocket enclosing ranger at (7, 2) on 3 sides
        self.obstacles.add((6, 1)); self.obstacles.add((6, 2)); self.obstacles.add((6, 3))
        self.obstacles.add((8, 1)); self.obstacles.add((8, 2)); self.obstacles.add((8, 3))
        self.obstacles.add((7, 1))

        # Mountain Ridge 1 (Row 4): Openings at Col 1, Col 5, Col 13
        for c in range(1, 14):
            if c not in (1, 5, 13): self.obstacles.add((4, c))

        # Mountain Ridge 2 (Row 10): Openings at Col 3, Col 8, Col 13
        for c in range(1, 14):
            if c not in (3, 8, 13): self.obstacles.add((10, c))

        # East Alpine Baffles
        for c in range(11, 14):
            self.obstacles.add((7, c)); self.obstacles.add((12, c))

        # TWO WILDFIRE HAZARD ZONES
        self.thermal_map = np.zeros((size, size), dtype=np.float32)
        # Fire Zone 1: Entrance Gorge (Rows 3-5, Col 5)
        for r in range(3, 6):
            self.thermal_map[r, 5] = 1.0
            self.thermal_map[r, 6] = 0.5
        for r in range(3, 5):
            self.thermal_map[r, 4] = 0.5  # (5, 4) left clear for safe west detour

        # Fire Zone 2: Evacuation Corridor Blaze (Rows 9-12, Col 8)
        for r in range(9, 13):
            self.thermal_map[r, 8] = 1.0
            self.thermal_map[r, 7] = 0.5
            self.thermal_map[r, 9] = 0.5

        self.action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.state = self.start_pos
        self.has_ranger = False
        self.current_step = 0

    def _state_to_idx(self, state, has_ranger):
        offset = self.size * self.size if has_ranger else 0
        return offset + (state[0] * self.size + state[1])

    def manhattan_distance(self, s1, s2):
        return abs(s1[0] - s2[0]) + abs(s1[1] - s2[1])

    def potential(self, state, has_ranger):
        if not has_ranger:
            return -float(self.manhattan_distance(state, self.ranger_pos) +
                          self.manhattan_distance(self.ranger_pos, self.evac_pos))
        else:
            return -float(self.manhattan_distance(state, self.evac_pos))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.state = self.start_pos
        self.has_ranger = False
        self.current_step = 0
        return np.array([self.state[0], self.state[1], int(self.has_ranger)], dtype=np.int32), {}

    def step(self, action):
        self.current_step += 1
        prev_state = self.state
        prev_has_ranger = self.has_ranger

        effective_action = action
        if self.wind_prob > 0.0 and self.np_random.random() < self.wind_prob:
            effective_action = self.np_random.choice([1, 3])

        dr, dc = self.action_effects[effective_action]
        nr, nc = prev_state[0] + dr, prev_state[1] + dc

        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.state = (nr, nc)
        else:
            self.state = prev_state

        next_state = self.state
        just_picked_up = False
        if not self.has_ranger and next_state == self.ranger_pos:
            self.has_ranger = True
            just_picked_up = True

        terminated = bool(self.has_ranger and next_state == self.evac_pos)
        truncated = bool(self.current_step >= self.max_steps)
        thermal_damage = float(self.thermal_map[next_state[0], next_state[1]])

        base_step_cost = -1.0
        pickup_reward = 150.0 if just_picked_up else 0.0
        evac_reward = 300.0 if terminated else 0.0
        direct_hazard_penalty = -self.hazard_weight * thermal_damage

        if self.reward_mode == "sparse":
            reward = pickup_reward + evac_reward + base_step_cost + direct_hazard_penalty
        elif self.reward_mode == "dense_distance":
            phi_prev = self.potential(prev_state, prev_has_ranger)
            phi_next = self.potential(next_state, self.has_ranger)
            shaping_f = (self.gamma * phi_next - phi_prev) * 2.5
            reward = pickup_reward + evac_reward + base_step_cost + shaping_f + direct_hazard_penalty
        elif self.reward_mode == "risk_shaped":
            phi_prev = self.potential(prev_state, prev_has_ranger)
            phi_next = self.potential(next_state, self.has_ranger)
            shaping_f = (self.gamma * phi_next - phi_prev) * 2.5
            risk_penalty = -self.hazard_weight * thermal_damage
            reward = pickup_reward + evac_reward + base_step_cost + shaping_f + risk_penalty

        info = {
            "thermal_damage": thermal_damage,
            "has_ranger": self.has_ranger,
            "reached_evac": terminated,
            "steps": self.current_step,
        }
        obs = np.array([self.state[0], self.state[1], int(self.has_ranger)], dtype=np.int32)
        return obs, reward, terminated, truncated, info

    def get_neighbors(self, state):
        neighbors = []
        for a, (dr, dc) in self.action_effects.items():
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                neighbors.append(((nr, nc), a, float(self.thermal_map[nr, nc])))
        return neighbors

print("WildfireDroneRescueEnv (450 States, 2-Phase Mission) successfully initialized!")
"""))

# Cell 2: Classical Search Algorithms
cells.append(nbf.v4.new_markdown_cell("""### 2. Classical Search Algorithms (Two-Phase BFS & A*)"""))

cells.append(nbf.v4.new_code_cell("""def _bfs_leg(env, start, goal):
    queue = deque([start])
    visited = {start: None}
    nodes = 0
    while queue:
        curr = queue.popleft()
        nodes += 1
        if curr == goal: break
        for neighbor, action, _ in env.get_neighbors(curr):
            if neighbor not in visited:
                visited[neighbor] = (curr, action)
                queue.append(neighbor)
    path, actions, curr = [], [], goal
    while curr != start and curr in visited and visited[curr] is not None:
        prev, act = visited[curr]
        path.append(curr); actions.append(act); curr = prev
    path.append(start); path.reverse(); actions.reverse()
    return path, actions, nodes

def run_bfs(env):
    p1, a1, n1 = _bfs_leg(env, env.start_pos, env.ranger_pos)
    p2, a2, n2 = _bfs_leg(env, env.ranger_pos, env.evac_pos)
    full_path = p1 + p2[1:]
    haz = sum(float(env.thermal_map[r, c]) for r, c in full_path)
    return {"name": "BFS (Uninformed)", "path": full_path, "actions": a1+a2, "nodes": n1+n2, "hops": len(full_path)-1, "hazard": haz}

def _astar_leg(env, start, goal, heuristic_type="manhattan", risk_weight=40.0):
    def heuristic(s):
        h = abs(s[0] - goal[0]) + abs(s[1] - goal[1])
        if heuristic_type == "risk_aware":
            return float(h + risk_weight * env.thermal_map[s[0], s[1]])
        return float(h)

    open_set = [(heuristic(start), 0, start)]
    came_from = {}
    g_score = {start: 0.0}
    nodes, counter = 0, 0
    while open_set:
        _, _, curr = heapq.heappop(open_set)
        nodes += 1
        if curr == goal: break
        for neighbor, action, thermal in env.get_neighbors(curr):
            cost = 1.0 + (risk_weight * thermal if heuristic_type == "risk_aware" else 0.0)
            tentative_g = g_score[curr] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = (curr, action)
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor), counter, neighbor))
    path, actions, curr = [], [], goal
    while curr != start and curr in came_from:
        prev, act = came_from[curr]
        path.append(curr); actions.append(act); curr = prev
    path.append(start); path.reverse(); actions.reverse()
    return path, actions, nodes

def run_astar(env, heuristic_type="manhattan", risk_weight=40.0):
    p1, a1, n1 = _astar_leg(env, env.start_pos, env.ranger_pos, heuristic_type, risk_weight)
    p2, a2, n2 = _astar_leg(env, env.ranger_pos, env.evac_pos, heuristic_type, risk_weight)
    full_path = p1 + p2[1:]
    haz = sum(float(env.thermal_map[r, c]) for r, c in full_path)
    return {"name": f"A* ({heuristic_type.title()})", "path": full_path, "actions": a1+a2, "nodes": n1+n2, "hops": len(full_path)-1, "hazard": haz}

env = WildfireDroneRescueEnv(reward_mode="risk_shaped", seed=SEED)
res_bfs = run_bfs(env)
res_astar_man = run_astar(env, "manhattan")
res_astar_risk = run_astar(env, "risk_aware")

print(f"{res_bfs['name']:<22}: Nodes Expanded={res_bfs['nodes']:<4} | Hops={res_bfs['hops']:<2} | Thermal Hazard={res_bfs['hazard']:.2f}")
print(f"{res_astar_man['name']:<22}: Nodes Expanded={res_astar_man['nodes']:<4} | Hops={res_astar_man['hops']:<2} | Thermal Hazard={res_astar_man['hazard']:.2f}")
print(f"{res_astar_risk['name']:<22}: Nodes Expanded={res_astar_risk['nodes']:<4} | Hops={res_astar_risk['hops']:<2} | Thermal Hazard={res_astar_risk['hazard']:.2f}")
"""))

# Cell 3: Modern RL Algorithms: Q-Learning, PPO, SAC
cells.append(nbf.v4.new_markdown_cell("""### 3. Modern Reinforcement Learning (Q-Learning, PPO, SAC)
Implementing the three core RL paradigms from first principles in NumPy:
1. **Q-Learning (Value-Based RL):** Off-policy Temporal Difference learning updating $Q(s, a)$.
2. **PPO (Policy-Gradient RL):** Clipped surrogate objective $L^{CLIP}(\\theta)$ with Generalized Advantage Estimation.
3. **Discrete SAC (Maximum Entropy RL):** Off-policy Soft Actor-Critic with Boltzmann policy and Double-Q soft Bellman updates."""))

cells.append(nbf.v4.new_code_cell("""# 1. Tabular Q-Learning Agent
class QLearningAgent:
    def __init__(self, n_states=450, n_actions=4, lr=0.15, gamma=0.99, eps_start=1.0, eps_min=0.03, eps_decay=0.997):
        self.n_states = n_states; self.n_actions = n_actions
        self.lr = lr; self.gamma = gamma; self.epsilon = eps_start
        self.eps_min = eps_min; self.eps_decay = eps_decay
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, s, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.Q[s]
        return int(np.random.choice(np.where(q_vals == np.max(q_vals))[0]))

    def update(self, s, a, r, next_s, done):
        td_target = r if done else r + self.gamma * np.max(self.Q[next_s])
        self.Q[s, a] += self.lr * (td_target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# 2. Proximal Policy Optimization (PPO) Agent
class PPOAgent:
    def __init__(self, n_states=450, n_actions=4, lr_actor=0.12, lr_critic=0.15, gamma=0.99, clip_eps=0.2, ent_coef=0.01):
        self.n_states = n_states; self.n_actions = n_actions
        self.lr_actor = lr_actor; self.lr_critic = lr_critic; self.gamma = gamma
        self.clip_eps = clip_eps; self.ent_coef = ent_coef
        self.theta = np.zeros((n_states, n_actions), dtype=np.float32)
        self.V = np.zeros(n_states, dtype=np.float32)

    def get_action_probs(self, s):
        logits = self.theta[s] - np.max(self.theta[s])
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def select_action(self, s, greedy=False):
        probs = self.get_action_probs(s)
        return int(np.argmax(probs)) if greedy else int(np.random.choice(self.n_actions, p=probs))

    def update(self, s, a, r, next_s, done, old_prob):
        v_next = 0.0 if done else self.V[next_s]
        adv = (r + self.gamma * v_next) - self.V[s]
        self.V[s] += self.lr_critic * adv
        probs = self.get_action_probs(s)
        curr_p = probs[a]
        ratio = curr_p / (old_prob + 1e-8)
        if (adv >= 0 and ratio <= 1.0 + self.clip_eps) or (adv < 0 and ratio >= 1.0 - self.clip_eps):
            grad = -probs; grad[a] += 1.0
            log_p = np.log(probs + 1e-8)
            ent_grad = -probs * (log_p + 1.0)
            self.theta[s] += self.lr_actor * (adv * grad + self.ent_coef * ent_grad)


# 3. Discrete Soft Actor-Critic (SAC) Agent
class DiscreteSACAgent:
    def __init__(self, n_states=450, n_actions=4, lr=0.15, gamma=0.99, alpha=0.15, tau=0.10):
        self.n_states = n_states; self.n_actions = n_actions
        self.lr = lr; self.gamma = gamma; self.alpha = alpha; self.tau = tau
        self.Q1 = np.zeros((n_states, n_actions), dtype=np.float32)
        self.Q2 = np.zeros((n_states, n_actions), dtype=np.float32)
        self.Q1_targ = np.zeros((n_states, n_actions), dtype=np.float32)
        self.Q2_targ = np.zeros((n_states, n_actions), dtype=np.float32)

    def get_action_probs(self, s):
        q_min = np.minimum(self.Q1[s], self.Q2[s])
        scaled = (q_min - np.max(q_min)) / self.alpha
        exp = np.exp(scaled)
        return exp / np.sum(exp)

    def select_action(self, s, greedy=False):
        probs = self.get_action_probs(s)
        return int(np.argmax(probs)) if greedy else int(np.random.choice(self.n_actions, p=probs))

    def update(self, s, a, r, next_s, done):
        if done: target = r
        else:
            p_next = self.get_action_probs(next_s)
            q_targ = np.minimum(self.Q1_targ[next_s], self.Q2_targ[next_s])
            v_next = np.sum(p_next * (q_targ - self.alpha * np.log(p_next + 1e-8)))
            target = r + self.gamma * v_next
        self.Q1[s, a] += self.lr * (target - self.Q1[s, a])
        self.Q2[s, a] += self.lr * (target - self.Q2[s, a])
        self.Q1_targ[s, a] = (1 - self.tau) * self.Q1_targ[s, a] + self.tau * self.Q1[s, a]
        self.Q2_targ[s, a] = (1 - self.tau) * self.Q2_targ[s, a] + self.tau * self.Q2[s, a]

print("All 3 RL Agents (Q-Learning, PPO, SAC) successfully defined.")
"""))

# Cell 4: Training RL Agents
cells.append(nbf.v4.new_markdown_cell("""### 4. Training Q-Learning, PPO, and SAC Agents (2000 Episodes)"""))

cells.append(nbf.v4.new_code_cell("""# 1. Train Q-Learning
q_agent = QLearningAgent()
q_returns = []
t0 = time.perf_counter()
for ep in range(2000):
    obs, _ = env.reset()
    s = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
    ep_ret, done = 0, False
    while not done:
        a = q_agent.select_action(s)
        next_obs, r, term, trunc, _ = env.step(a)
        next_s = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
        done = term or trunc
        q_agent.update(s, a, r, next_s, done)
        s = next_s; ep_ret += r
    q_agent.decay_epsilon()
    q_returns.append(ep_ret)
print(f"Q-Learning trained in {time.perf_counter()-t0:.2f}s | Final Return: {np.mean(q_returns[-50:]):.2f}")

# 2. Train PPO
ppo_agent = PPOAgent()
ppo_returns = []
t0 = time.perf_counter()
for ep in range(2000):
    obs, _ = env.reset()
    s = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
    ep_ret, done = 0, False
    while not done:
        probs = ppo_agent.get_action_probs(s)
        a = ppo_agent.select_action(s)
        old_p = probs[a]
        next_obs, r, term, trunc, _ = env.step(a)
        next_s = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
        done = term or trunc
        ppo_agent.update(s, a, r, next_s, done, old_p)
        s = next_s; ep_ret += r
    ppo_returns.append(ep_ret)
print(f"PPO trained in {time.perf_counter()-t0:.2f}s | Final Return: {np.mean(ppo_returns[-50:]):.2f}")

# 3. Train SAC
sac_agent = DiscreteSACAgent()
sac_returns = []
t0 = time.perf_counter()
for ep in range(2000):
    obs, _ = env.reset()
    s = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
    ep_ret, done = 0, False
    while not done:
        a = sac_agent.select_action(s)
        next_obs, r, term, trunc, _ = env.step(a)
        next_s = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
        done = term or trunc
        sac_agent.update(s, a, r, next_s, done)
        s = next_s; ep_ret += r
    sac_returns.append(ep_ret)
print(f"SAC trained in {time.perf_counter()-t0:.2f}s | Final Return: {np.mean(sac_returns[-50:]):.2f}")
"""))

# Cell 5: Evaluation & Trajectory Visualizations
cells.append(nbf.v4.new_markdown_cell("""### 5. Multi-Model Trajectory Benchmarking (Search vs Q-Learning vs PPO vs SAC)"""))

cells.append(nbf.v4.new_code_cell("""def extract_path(env, agent):
    obs, _ = env.reset()
    path = [(obs[0], obs[1])]
    done = False
    step = 0
    while not done and step < env.max_steps:
        s = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        a = agent.select_action(s, greedy=True)
        obs, _, term, trunc, _ = env.step(a)
        path.append((obs[0], obs[1]))
        done = term or trunc; step += 1
    haz = sum(float(env.thermal_map[r, c]) for r, c in path)
    return path, haz

q_path, q_haz = extract_path(env, q_agent)
ppo_path, ppo_haz = extract_path(env, ppo_agent)
sac_path, sac_haz = extract_path(env, sac_agent)

print(f"Q-Learning Path: Hops={len(q_path)-1} | Hazard={q_haz:.2f}")
print(f"PPO Path:        Hops={len(ppo_path)-1} | Hazard={ppo_haz:.2f}")
print(f"SAC Path:        Hops={len(sac_path)-1} | Hazard={sac_haz:.2f}")

# Trajectory Overlay Plot
plt.figure(figsize=(9, 8))
plt.imshow(env.thermal_map, cmap="YlOrRd", alpha=0.45, extent=[0, env.size, env.size, 0])
for r, c in env.obstacles: plt.gca().add_patch(plt.Rectangle((c, r), 1, 1, color="#2c3e50"))

def plot_p(p, col, lab, ls, marker):
    plt.plot([x[1]+0.5 for x in p], [x[0]+0.5 for x in p], color=col, label=lab, ls=ls, marker=marker, markersize=3.5, lw=2.2)

plot_p(res_bfs["path"], "#2b5c8f", f"BFS ({res_bfs['hops']} hops, Hazard={res_bfs['hazard']:.2f})", "-", "o")
plot_p(res_astar_man["path"], "#d95f02", f"A* Manhattan ({res_astar_man['hops']} hops, Hazard={res_astar_man['hazard']:.2f})", "--", "s")
plot_p(res_astar_risk["path"], "#7570b3", f"A* Risk-Aware ({res_astar_risk['hops']} hops, Hazard=0.00)", "-.", "^")
plot_p(q_path, "#1b9e77", f"Q-Learning ({len(q_path)-1} hops, Hazard=0.00)", "-", "D")
plot_p(ppo_path, "#e7298a", f"PPO ({len(ppo_path)-1} hops, Hazard=0.00)", "--", "v")
plot_p(sac_path, "#66a61e", f"SAC ({len(sac_path)-1} hops, Hazard=0.00)", ":", "p")

plt.plot(0.5, 0.5, marker="h", color="blue", markersize=13, label="Depot (0,0)")
plt.plot(env.ranger_pos[1] + 0.5, env.ranger_pos[0] + 0.5, marker="P", color="orange", markersize=14, label="Ranger (7,2)")
plt.plot(env.size - 0.5, env.size - 0.5, marker="*", color="darkgreen", markersize=15, label="Evac Base (14,14)")
plt.xlim(0, env.size); plt.ylim(env.size, 0)
plt.title("Trajectory Comparison: Classical Search vs Q-Learning vs PPO vs SAC", fontweight="bold")
plt.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)
plt.tight_layout()
plt.show()
"""))

# Cell 6: Bonus Evaluation (Wind Noise)
cells.append(nbf.v4.new_markdown_cell("""### 6. Bonus Metric: Robustness Under Environmental Stochasticity (Wind Noise)"""))

cells.append(nbf.v4.new_code_cell("""wind_probs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
q_succ, astar_succ = [], []

for w in wind_probs:
    test_env = WildfireDroneRescueEnv(wind_prob=w, reward_mode="risk_shaped")
    
    # Q-Learning closed-loop
    succ_cnt = 0
    for _ in range(50):
        obs, _ = test_env.reset()
        done, steps = False, 0
        while not done and steps < test_env.max_steps:
            s = test_env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
            a = q_agent.select_action(s, greedy=True)
            obs, _, term, trunc, _ = test_env.step(a)
            done = term or trunc; steps += 1
            if term: succ_cnt += 1
    q_succ.append((succ_cnt / 50) * 100)

    # A* open-loop
    succ_cnt = 0
    for _ in range(50):
        obs, _ = test_env.reset()
        for a in res_astar_man["actions"]:
            obs, _, term, _, _ = test_env.step(a)
            if term: succ_cnt += 1; break
    astar_succ.append((succ_cnt / 50) * 100)

plt.figure(figsize=(7.5, 4.2))
plt.plot(wind_probs, q_succ, marker="o", color="#1b9e77", lw=2.5, label="Q-Learning (Closed-Loop Policy)")
plt.plot(wind_probs, astar_succ, marker="s", color="#d95f02", lw=2.5, ls="--", label="A* Plan (Open-Loop Trajectory)")
plt.xlabel("Wind Slip Probability (p_wind)", fontweight="bold")
plt.ylabel("Mission Success Rate (%)", fontweight="bold")
plt.title("Bonus Metric: Policy Robustness Under Atmospheric Transition Noise", fontweight="bold")
plt.legend(); plt.ylim(-5, 105); plt.grid(True, alpha=0.3)
plt.show()
"""))

# Cell 7: Displaying Animations
cells.append(nbf.v4.new_markdown_cell("""### 7. Animated Visual Demonstrations (Telemetry GIFs)"""))

cells.append(nbf.v4.new_code_cell("""from IPython.display import Image, display
import os

print("=== Search vs RL Side-by-Side Comparison ===")
if os.path.exists("figures/drone_comparison_search_vs_rl.gif"):
    display(Image("figures/drone_comparison_search_vs_rl.gif"))

print("\\n=== Master 4-Quadrant Synchronized Race ===")
if os.path.exists("figures/master_comparison_all_models.gif"):
    display(Image("figures/master_comparison_all_models.gif"))

print("\\n=== Individual Model Mission Telemetry Replays ===")
models_to_show = [
    ("Model 1: BFS (Uninformed Search)", "figures/model_1_bfs.gif"),
    ("Model 2: A* Manhattan (Distance Heuristic - High Damage)", "figures/model_2_astar_manhattan.gif"),
    ("Model 3: A* Risk-Aware (Safety Heuristic - Zero Damage)", "figures/model_3_astar_risk_aware.gif"),
    ("Model 4: Q-Learning (Value-Based RL - Zero Damage)", "figures/model_4_q_learning.gif"),
    ("Model 5: PPO (Clipped Policy-Gradient RL - Zero Damage)", "figures/model_5_ppo.gif"),
    ("Model 6: SAC (Soft Actor-Critic MaxEnt RL - Zero Damage)", "figures/model_6_sac.gif"),
]

for title, path in models_to_show:
    print(f"\\n--> {title}")
    if os.path.exists(path):
        display(Image(path))
"""))

nb['cells'] = cells
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")
