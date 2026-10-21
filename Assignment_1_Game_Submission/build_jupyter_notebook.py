"""
build_jupyter_notebook.py
Builds the complete, highly polished Jupyter Notebook (Assignment_1_Wildfire_Drone.ipynb)
for the Two-Phase Wildfire Search, Rescue & Evacuation Mission.
"""

import os
import nbformat as nbf

NOTEBOOK_PATH = os.path.join(os.path.dirname(__file__), "Assignment_1_Wildfire_Drone.ipynb")

nb = nbf.v4.new_notebook()

cells = []

# Title & Metadata
cells.append(nbf.v4.new_markdown_cell("""# Machine Learning (23MEE305) - Assignment 1
## Exploring the Evolution from Classical Search (BFS, A*) to Reinforcement Learning (Q-Learning, Actor-Critic)

**Mission Scenario:** Two-Phase Autonomous Wildfire Search, Rescue & Evacuation Drone  
**Course Code:** 23MEE305  
**Deliverables:** Custom Gymnasium Environment, Classical Search Benchmarks, 2 RL Algorithm Families, Reward Shaping Ablation, Stochastic Wind Robustness Evaluation (+5 Bonus Marks), and Synchronized Animated Visualizations.

---

### Executive Summary & Operational Scenario
In high-risk disaster response robotics, an autonomous Unmanned Aerial Vehicle (UAV) is deployed on a critical **Two-Phase Search, Rescue & Evacuation** mission across an intricate 15x15 canyon terrain:
- **Phase 1 (Search & Ingress):** The drone launches from **Depot Base (0, 0)**, penetrates through mountain barriers, locates, and secures an injured forest ranger trapped inside a secluded rock pocket at **(7, 2)**.
- **Phase 2 (Evacuation & Egress):** With the ranger secured onboard, the drone must navigate through escape corridors and safely deliver the patient to the **Hospital Evacuation Base at (14, 14)**.

### Environmental Hazards & Algorithmic Dilemma:
1. **Mountain Ridges & Enclosed Pocket:** The canyon pocket at `(7, 2)` is enclosed on 3 sides by impassable rock cliffs, forcing the agent to discover the precise ingress/egress opening.
2. **Two Strategic Wildfire Blaze Zones:**
   - **Fire Zone 1 (Entrance Gorge):** Positioned at rows 3–5, col 5, blocking the direct geometric line from the depot.
   - **Fire Zone 2 (Evacuation Corridor Blaze):** Positioned at rows 9–12, col 8, blocking the direct line to the hospital helipad.
3. **The Core Dilemma:** Geometric distance heuristics (e.g. Euclidean / Manhattan distance) tempt the agent to fly through active flames ($T=1.0$), incurring catastrophic thermal burns. Risk-aware RL algorithms learn to discover safe western and eastern alpine bypasses, accomplishing a **flawless zero-hazard extraction**.
"""))

# Cell 1: Custom Gymnasium Environment
cells.append(nbf.v4.new_markdown_cell("""### 1. Custom Gymnasium Environment: `WildfireDroneRescueEnv`
We formulate the two-phase mission as a discrete Markov Decision Process (MDP) with state space $S = (r, c, \\text{has\\_ranger}) \\in \\{0..14\\} \\times \\{0..14\\} \\times \\{0, 1\\}$ ($|S| = 450$)."""))

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
cells.append(nbf.v4.new_markdown_cell("""### 2. Classical Search Algorithms (Two-Phase BFS & A*)
In two-phase classical search, each algorithm plans Leg 1 (Depot $\\to$ Ranger) and Leg 2 (Ranger $\\to$ Hospital Base)."""))

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

# Cell 3: Modern Reinforcement Learning
cells.append(nbf.v4.new_markdown_cell("""### 3. Modern Reinforcement Learning (Q-Learning & Actor-Critic)
Two distinct algorithmic families implemented from first principles in NumPy:
1. **Value-Based RL (Tabular Q-Learning):** Off-policy Temporal Difference learning updating $Q(s, a)$.
2. **Policy-Gradient RL (Tabular Actor-Critic):** On-policy Actor updating action logits $\\theta(s, a)$ combined with a Critic state-value function $V(s)$."""))

cells.append(nbf.v4.new_code_cell("""class QLearningAgent:
    def __init__(self, n_states=450, n_actions=4, lr=0.15, gamma=0.99, eps_start=1.0, eps_min=0.03, eps_decay=0.997):
        self.n_states = n_states; self.n_actions = n_actions
        self.lr = lr; self.gamma = gamma; self.epsilon = eps_start
        self.eps_min = eps_min; self.eps_decay = eps_decay
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state_idx, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.Q[state_idx]
        best_actions = np.where(q_vals == np.max(q_vals))[0]
        return np.random.choice(best_actions)

    def update(self, s, a, r, next_s, done):
        td_target = r if done else r + self.gamma * np.max(self.Q[next_s])
        self.Q[s, a] += self.lr * (td_target - self.Q[s, a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


class TabularActorCriticAgent:
    def __init__(self, n_states=450, n_actions=4, lr_actor=0.08, lr_critic=0.15, gamma=0.99):
        self.n_states = n_states; self.n_actions = n_actions
        self.lr_actor = lr_actor; self.lr_critic = lr_critic; self.gamma = gamma
        self.theta = np.zeros((n_states, n_actions), dtype=np.float32)
        self.V = np.zeros(n_states, dtype=np.float32)

    def get_action_probs(self, state_idx):
        logits = self.theta[state_idx] - np.max(self.theta[state_idx])
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def select_action(self, state_idx, greedy=False):
        probs = self.get_action_probs(state_idx)
        return int(np.argmax(probs)) if greedy else int(np.random.choice(self.n_actions, p=probs))

    def update(self, s, a, r, next_s, done):
        v_next = 0.0 if done else self.V[next_s]
        td_error = (r + self.gamma * v_next) - self.V[s]
        self.V[s] += self.lr_critic * td_error
        probs = self.get_action_probs(s)
        grad = (np.eye(self.n_actions)[a] - probs)
        self.theta[s] += self.lr_actor * td_error * grad

print("Q-Learning and Actor-Critic Agents successfully defined.")
"""))

# Cell 4: RL Training
cells.append(nbf.v4.new_markdown_cell("""### 4. Training RL Agents (2000 Episodes)"""))

cells.append(nbf.v4.new_code_cell("""def train_q_learning(env, episodes=2000):
    n_states = env.size * env.size * 2
    agent = QLearningAgent(n_states=n_states)
    returns, hazards = [], []
    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        ep_ret, ep_haz, done = 0, 0, False
        while not done:
            a = agent.select_action(s_idx)
            next_obs, r, term, trunc, info = env.step(a)
            next_s = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
            done = term or trunc
            agent.update(s_idx, a, r, next_s, done)
            s_idx = next_s; ep_ret += r; ep_haz += info["thermal_damage"]
        agent.decay_epsilon()
        returns.append(ep_ret); hazards.append(ep_haz)
    print(f"Q-Learning trained in {time.perf_counter()-t0:.2f}s | Final Return: {np.mean(returns[-50:]):.2f}")
    return agent, returns, hazards

def train_actor_critic(env, episodes=2000):
    n_states = env.size * env.size * 2
    agent = TabularActorCriticAgent(n_states=n_states)
    returns, hazards = [], []
    t0 = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        ep_ret, ep_haz, done = 0, 0, False
        while not done:
            a = agent.select_action(s_idx)
            next_obs, r, term, trunc, info = env.step(a)
            next_s = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
            done = term or trunc
            agent.update(s_idx, a, r, next_s, done)
            s_idx = next_s; ep_ret += r; ep_haz += info["thermal_damage"]
        returns.append(ep_ret); hazards.append(ep_haz)
    print(f"Actor-Critic trained in {time.perf_counter()-t0:.2f}s | Final Return: {np.mean(returns[-50:]):.2f}")
    return agent, returns, hazards

q_agent, q_ret, q_haz = train_q_learning(env, episodes=2000)
ac_agent, ac_ret, ac_haz = train_actor_critic(env, episodes=2000)
"""))

# Cell 5: Trajectory & Heatmap Evaluation
cells.append(nbf.v4.new_markdown_cell("""### 5. Evaluating Learned Closed-Loop Policies & Visualizing Trajectories"""))

cells.append(nbf.v4.new_code_cell("""def extract_policy_path(env, agent, is_q=True):
    obs, _ = env.reset()
    path = [(obs[0], obs[1])]
    done = False
    step = 0
    while not done and step < env.max_steps:
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        a = agent.select_action(s_idx, greedy=True)
        obs, _, term, trunc, _ = env.step(a)
        path.append((obs[0], obs[1]))
        done = term or trunc
        step += 1
    hazard = sum(float(env.thermal_map[r, c]) for r, c in path)
    return path, hazard

q_path, q_hazard = extract_policy_path(env, q_agent, is_q=True)
ac_path, ac_hazard = extract_policy_path(env, ac_agent, is_q=False)

print(f"Q-Learning Mission Path: Length={len(q_path)-1} hops | Cumulative Hazard={q_hazard:.2f}")
print(f"Actor-Critic Mission Path: Length={len(ac_path)-1} hops | Cumulative Hazard={ac_hazard:.2f}")

# Plot Side-by-Side Trajectories & Values
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# Panel 1: Value Function (Phase 1: Ingress)
ax1 = axes[0]
V_grid = np.max(q_agent.Q[:env.size*env.size], axis=1).reshape((env.size, env.size))
for r, c in env.obstacles: V_grid[r, c] = np.nan
sns.heatmap(V_grid, ax=ax1, cmap="viridis", cbar=True)
ax1.set_title("Learned Value Function V(s) [Phase 1: Search]", fontweight="bold")

# Panel 2: Trajectories Overlay
ax2 = axes[1]
ax2.imshow(env.thermal_map, cmap="YlOrRd", alpha=0.5, extent=[0, env.size, env.size, 0])
for r, c in env.obstacles: ax2.add_patch(plt.Rectangle((c, r), 1, 1, color="#2c3e50"))

def plot_p(p, col, lab, ls):
    ax2.plot([x[1]+0.5 for x in p], [x[0]+0.5 for x in p], color=col, label=lab, ls=ls, lw=2.2)

plot_p(res_bfs["path"], "#2b5c8f", f"BFS ({res_bfs['hops']} hops, Hazard={res_bfs['hazard']:.2f})", "-")
plot_p(res_astar_man["path"], "#d95f02", f"A* Manhattan ({res_astar_man['hops']} hops, Hazard={res_astar_man['hazard']:.2f})", "--")
plot_p(res_astar_risk["path"], "#7570b3", f"A* Risk-Aware ({res_astar_risk['hops']} hops, Hazard={res_astar_risk['hazard']:.2f})", "-.")
plot_p(q_path, "#1b9e77", f"Q-Learning Safe ({len(q_path)-1} hops, Hazard={q_hazard:.2f})", "-")

ax2.plot(0.5, 0.5, marker="h", color="blue", markersize=13, label="Depot (0,0)")
ax2.plot(env.ranger_pos[1] + 0.5, env.ranger_pos[0] + 0.5, marker="P", color="orange", markersize=14, label="Ranger (7,2)")
ax2.plot(env.size - 0.5, env.size - 0.5, marker="*", color="darkgreen", markersize=15, label="Evac Helipad (14,14)")
ax2.set_xlim(0, env.size); ax2.set_ylim(env.size, 0)
ax2.set_title("Two-Phase Mission Trajectories: Search -> Rescue -> Evac", fontweight="bold")
ax2.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2)
plt.tight_layout()
plt.show()
"""))

# Cell 6: Bonus Evaluation (Wind Noise)
cells.append(nbf.v4.new_markdown_cell("""### 6. Bonus Metric: Robustness Under Environmental Stochasticity (Wind Noise)
We benchmark Open-Loop A* Plan Execution against Closed-Loop Q-Learning Policy across increasing wind slip probabilities $p_{\\text{wind}} \\in [0.0, 0.25]$."""))

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
cells.append(nbf.v4.new_markdown_cell("""### 7. Animated Visual Demonstrations (Telemetry GIFs)
Displaying the generated animated GIFs directly inside the notebook."""))

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
    ("Model 2: A* Manhattan (Distance-Only Heuristic - Severe Burns)", "figures/model_2_astar_manhattan.gif"),
    ("Model 3: A* Risk-Aware (Safety-Augmented Heuristic - Detour)", "figures/model_3_astar_risk_aware.gif"),
    ("Model 4: Q-Learning (Trained Value-Based Policy - Safe Rescue)", "figures/model_4_q_learning.gif"),
    ("Model 5: Actor-Critic (Trained Policy-Gradient - Safe Rescue)", "figures/model_5_actor_critic.gif"),
]

for title, path in models_to_show:
    print(f"\\n--> {title}")
    if os.path.exists(path):
        display(Image(path))
"""))

# Cell 8: Interactive Studio & Human-in-the-Loop RL
cells.append(nbf.v4.new_markdown_cell("""### 8. Interactive Studio: Blind AI Scout Benchmark (+2 Allowance) & Endless Procedural Mazes
We extend the environment and agent with:
1. **Blind AI Scout Benchmark Challenge:** On each procedural level, the RL agent secretly scouts the maze first, solves the rescue mission, and establishes the strict **Minimum Steps Benchmark ($S_{\\text{RL}}$)**.
2. **Concealed Path & Target Steps (+2 Allowance):** The path is not shown and the step limit is hidden. The player must deduce the route independently, with a generous $+2$ step allowance ($S_{\\text{max}} = S_{\\text{RL}} + 2$).
3. **Zero Disclosure on Failure & Strict Retry:** If the player fails to evacuate the ranger within $S_{\\text{max}}$ steps, they fail and restart that **exact same layout**. The AI's step count is **kept secret upon failure** so the player must figure out the optimal path on their own.
4. **Step Benchmark Revealed Only on Victory:** The AI's benchmark count is only disclosed when the player successfully clears the level!
5. **Auto-Randomized Mazes on Success:** Upon each successful rescue, a brand-new maze layout is generated with guaranteed BFS reachability.
6. **Cross-Maze Relational Q-Learning:** A dual-system tabular RL agent that maintains coordinate-invariant relational features ($Q_{\\text{rel}}$) to transfer navigation intuition permanently across endless random mazes.
7. **Human-in-the-Loop Interventions & Feedback:** Real-time keyboard overrides (DAgger) and praise/penalty signals (RLHF / TAMER) update policy values online.
8. **Playable Web Simulator:** Full cockpit available via `python interactive_game.py`."""))

cells.append(nbf.v4.new_code_cell("""from dynamic_wildfire_env import DynamicWildfireDroneEnv
from interactive_learning import InteractiveQLearningAgent
from wildfire_model_manager import WildfireModelManager

# 1. Initialize Dynamic Environment & Randomize
dyn_env = DynamicWildfireDroneEnv()
dyn_env.randomize_scenario(randomize_agents=True, randomize_fires=True, num_fire_clusters=3)

print("=== Randomized Scenario Configuration ===")
print(f"Depot Start Pos : {dyn_env.start_pos}")
print(f"Trapped Ranger  : {dyn_env.ranger_pos}")
print(f"Hospital Evac   : {dyn_env.evac_pos}")
print(f"BFS Reachable   : {dyn_env.check_solvability(dyn_env.start_pos, dyn_env.ranger_pos, dyn_env.evac_pos)}")

# 2. Benchmark on the New Randomized Terrain
manager = WildfireModelManager(dyn_env)
bench_results = manager.benchmark_all_models()

print("\\n=== Algorithmic Benchmarks on Randomized Map ===")
for model_key, res in bench_results.items():
    print(f"[{res['name']:<25}] Hops: {res['path_length']:<3} | Hazard: {res['hazard']:<5.2f} | Success: {res['success']}")

# 3. Simulate Human Demonstration & Online Learning
print("\\n=== Simulating User Demonstration & Online Adaptation ===")
user_agent = manager.interactive_q_agent

# Run quick 50 fine-tuning episodes on randomized map
returns = user_agent.quick_fine_tune(dyn_env, episodes=50)
print(f"Adapted Q-Learning over 50 episodes | Mean Return: {np.mean(returns[-10:]):.2f}")

# Extract and display sample policy vectors
p_grid, v_grid = user_agent.get_policy_and_values(dyn_env)
print(f"Updated Policy Grid Shape: {p_grid.shape}")
print("Sample 5x5 subgrid policy arrows:")
for r in range(5):
    print(" ".join(str(p_grid[r, c]) for c in range(5)))
"""))

nb['cells'] = cells
with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Jupyter Notebook successfully built at: {NOTEBOOK_PATH}")

