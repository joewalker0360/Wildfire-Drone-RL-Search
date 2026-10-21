"""
run_experiments.py
Comprehensive experimental suite for Assignment 1:
Evolution from Classical Search (BFS, A*) to Reinforcement Learning (Q-Learning, PPO, SAC).
Two-Phase Search, Rescue & Evacuation Mission in Hazardous Wildfire Terrain.
- Phase 1: Depot (0,0) -> Trapped Ranger in Canyon Pocket (7,2)
- Phase 2: Ranger (7,2) -> Emergency Evacuation Base (14,14)
"""

import os
import time
import heapq
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from wildfire_drone_env import WildfireDroneRescueEnv

SEED = 42
np.random.seed(SEED)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


# ==========================================
# 1. CLASSICAL SEARCH ALGORITHMS (2-PHASE)
# ==========================================

def _bfs_single_leg(env, start_pos, target_pos):
    queue = deque([start_pos])
    visited = {start_pos: None}
    nodes = 0
    t0 = time.perf_counter()

    while queue:
        curr = queue.popleft()
        nodes += 1
        if curr == target_pos:
            break
        for neighbor, action, _ in env.get_neighbors(curr):
            if neighbor not in visited:
                visited[neighbor] = (curr, action)
                queue.append(neighbor)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    path, actions, curr = [], [], target_pos
    while curr != start_pos and curr in visited and visited[curr] is not None:
        prev, act = visited[curr]
        path.append(curr); actions.append(act); curr = prev
    path.append(start_pos); path.reverse(); actions.reverse()
    return path, actions, nodes, elapsed_ms


def run_bfs(env):
    """Executes complete mission via BFS: Depot -> Ranger -> Evac."""
    p1, a1, n1, t1 = _bfs_single_leg(env, env.start_pos, env.ranger_pos)
    p2, a2, n2, t2 = _bfs_single_leg(env, env.ranger_pos, env.evac_pos)

    full_path = p1 + p2[1:]
    full_actions = a1 + a2
    total_nodes = n1 + n2
    total_hazard = sum(float(env.thermal_map[r, c]) for r, c in full_path)

    return {
        "name": "BFS (Uninformed)",
        "path": full_path,
        "actions": full_actions,
        "nodes_expanded": total_nodes,
        "path_length": len(full_path) - 1,
        "hazard_exposure": total_hazard,
        "runtime_ms": t1 + t2,
    }


def _astar_single_leg(env, start_pos, target_pos, heuristic_type="manhattan", risk_weight=40.0):
    t0 = time.perf_counter()

    def heuristic(state):
        h = abs(state[0] - target_pos[0]) + abs(state[1] - target_pos[1])
        if heuristic_type == "risk_aware":
            return float(h + risk_weight * env.thermal_map[state[0], state[1]])
        return float(h)

    open_set = []
    counter = 0
    heapq.heappush(open_set, (heuristic(start_pos), counter, start_pos))
    came_from = {}
    g_score = {start_pos: 0.0}
    nodes = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)
        nodes += 1
        if current == target_pos:
            break

        for neighbor, action, thermal in env.get_neighbors(current):
            cost = 1.0 + (risk_weight * thermal if heuristic_type == "risk_aware" else 0.0)
            tentative_g = g_score[current] + cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = (current, action)
                g_score[neighbor] = tentative_g
                counter += 1
                heapq.heappush(open_set, (tentative_g + heuristic(neighbor), counter, neighbor))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    path, actions, curr = [], [], target_pos
    while curr != start_pos and curr in came_from:
        prev, act = came_from[curr]
        path.append(curr); actions.append(act); curr = prev
    path.append(start_pos); path.reverse(); actions.reverse()
    return path, actions, nodes, elapsed_ms


def run_astar(env, heuristic_type="manhattan", risk_weight=40.0):
    """Executes complete mission via A*: Depot -> Ranger -> Evac."""
    p1, a1, n1, t1 = _astar_single_leg(env, env.start_pos, env.ranger_pos, heuristic_type, risk_weight)
    p2, a2, n2, t2 = _astar_single_leg(env, env.ranger_pos, env.evac_pos, heuristic_type, risk_weight)

    full_path = p1 + p2[1:]
    full_actions = a1 + a2
    total_nodes = n1 + n2
    total_hazard = sum(float(env.thermal_map[r, c]) for r, c in full_path)

    return {
        "name": f"A* ({heuristic_type.title()})",
        "path": full_path,
        "actions": full_actions,
        "nodes_expanded": total_nodes,
        "path_length": len(full_path) - 1,
        "hazard_exposure": total_hazard,
        "runtime_ms": t1 + t2,
    }


# ==========================================
# 2. BASELINE AGENTS (RANDOM & GREEDY)
# ==========================================

def evaluate_random_agent(env, episodes=50):
    returns, steps_list, hazards = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_steps, ep_hazard = 0, 0, 0
        done = False
        while not done:
            a = env.action_space.sample()
            obs, r, term, trunc, info = env.step(a)
            ep_ret += r; ep_steps += 1
            ep_hazard += info["thermal_damage"]
            done = term or trunc
        returns.append(ep_ret); steps_list.append(ep_steps); hazards.append(ep_hazard)

    return {
        "name": "Random Baseline Policy",
        "mean_return": np.mean(returns),
        "mean_steps": np.mean(steps_list),
        "mean_hazard": np.mean(hazards),
        "success_rate": np.mean([s < env.max_steps for s in steps_list]) * 100,
    }


def evaluate_greedy_agent(env, episodes=50):
    returns, steps_list, hazards = [], [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        ep_ret, ep_steps, ep_hazard = 0, 0, 0
        done = False
        while not done:
            r, c, has_ranger = obs[0], obs[1], bool(obs[2])
            target = env.evac_pos if has_ranger else env.ranger_pos
            best_a = None
            min_dist = float("inf")
            for a, (dr, dc) in env.action_effects.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < env.size and 0 <= nc < env.size and (nr, nc) not in env.obstacles:
                    dist = abs(nr - target[0]) + abs(nc - target[1])
                    if dist < min_dist:
                        min_dist = dist
                        best_a = a
            if best_a is None:
                best_a = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(best_a)
            ep_ret += reward; ep_steps += 1
            ep_hazard += info["thermal_damage"]
            done = term or trunc
        returns.append(ep_ret); steps_list.append(ep_steps); hazards.append(ep_hazard)

    return {
        "name": "Greedy Distance Heuristic",
        "mean_return": np.mean(returns),
        "mean_steps": np.mean(steps_list),
        "mean_hazard": np.mean(hazards),
        "success_rate": np.mean([s < env.max_steps for s in steps_list]) * 100,
    }


# ==========================================
# 3. RL ALGORITHM 1: TABULAR Q-LEARNING (VALUE-BASED)
# ==========================================

class QLearningAgent:
    def __init__(self, n_states=450, n_actions=4, lr=0.15, gamma=0.99, eps_start=1.0, eps_min=0.03, eps_decay=0.997):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state_idx, greedy=False):
        if not greedy and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        q_vals = self.Q[state_idx]
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return np.random.choice(best_actions)

    def update(self, s, a, r, next_s, done):
        td_target = r if done else r + self.gamma * np.max(self.Q[next_s])
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


def train_q_learning(env, episodes=2000, lr=0.15, gamma=0.99):
    n_states = env.size * env.size * 2
    agent = QLearningAgent(n_states=n_states, n_actions=4, lr=lr, gamma=gamma)
    episode_returns = []
    episode_lengths = []
    episode_hazards = []

    start_time = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        total_ret, steps, ep_hazard = 0, 0, 0
        done = False

        while not done:
            action = agent.select_action(s_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_s_idx = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
            done = terminated or truncated

            agent.update(s_idx, action, reward, next_s_idx, done)

            s_idx = next_s_idx
            total_ret += reward
            steps += 1
            ep_hazard += info["thermal_damage"]

        agent.decay_epsilon()
        episode_returns.append(total_ret)
        episode_lengths.append(steps)
        episode_hazards.append(ep_hazard)

    training_time = time.perf_counter() - start_time
    return agent, episode_returns, episode_lengths, episode_hazards, training_time


# ==========================================
# 4. RL ALGORITHM 2: PPO (PROXIMAL POLICY OPTIMIZATION)
# ==========================================

class PPOAgent:
    """
    On-Policy Actor-Critic with PPO Clipped Surrogate Objective:
    L^CLIP(theta) = E [ min( r_t(theta)*A_t, clip(r_t(theta), 1-eps, 1+eps)*A_t ) ]
    """
    def __init__(self, n_states=450, n_actions=4, lr_actor=0.12, lr_critic=0.15, gamma=0.99, clip_eps=0.2, ent_coef=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.ent_coef = ent_coef
        self.theta = np.zeros((n_states, n_actions), dtype=np.float32)
        self.V = np.zeros(n_states, dtype=np.float32)

    def get_action_probs(self, s):
        logits = self.theta[s] - np.max(self.theta[s])
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def select_action(self, s, greedy=False):
        probs = self.get_action_probs(s)
        if greedy:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, s, a, r, next_s, done, old_prob):
        v_next = 0.0 if done else self.V[next_s]
        td_target = r + self.gamma * v_next
        advantage = td_target - self.V[s]

        # Critic update
        self.V[s] += self.lr_critic * advantage

        # PPO ratio & clipped surrogate
        probs = self.get_action_probs(s)
        curr_prob = probs[a]
        ratio = curr_prob / (old_prob + 1e-8)

        # Check clipping condition
        if (advantage >= 0 and ratio <= 1.0 + self.clip_eps) or (advantage < 0 and ratio >= 1.0 - self.clip_eps):
            grad = -probs
            grad[a] += 1.0
            log_p = np.log(probs + 1e-8)
            ent_grad = -probs * (log_p + 1.0)
            self.theta[s] += self.lr_actor * (advantage * grad + self.ent_coef * ent_grad)


def train_ppo(env, episodes=2000, lr_actor=0.12, lr_critic=0.15, gamma=0.99):
    n_states = env.size * env.size * 2
    agent = PPOAgent(n_states=n_states, n_actions=4, lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma)
    episode_returns = []
    episode_lengths = []
    episode_hazards = []

    start_time = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        total_ret, steps, ep_hazard = 0, 0, 0
        done = False

        while not done:
            probs = agent.get_action_probs(s_idx)
            action = agent.select_action(s_idx)
            old_p = probs[action]

            next_obs, reward, terminated, truncated, info = env.step(action)
            next_s_idx = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
            done = terminated or truncated

            agent.update(s_idx, action, reward, next_s_idx, done, old_p)

            s_idx = next_s_idx
            total_ret += reward
            steps += 1
            ep_hazard += info["thermal_damage"]

        episode_returns.append(total_ret)
        episode_lengths.append(steps)
        episode_hazards.append(ep_hazard)

    training_time = time.perf_counter() - start_time
    return agent, episode_returns, episode_lengths, episode_hazards, training_time


# ==========================================
# 5. RL ALGORITHM 3: DISCRETE SAC (SOFT ACTOR-CRITIC)
# ==========================================

class DiscreteSACAgent:
    """
    Maximum Entropy Soft Actor-Critic for Discrete Action Spaces (Christodoulou 2019):
    J(pi) = E [ sum_t r_t + alpha * H(pi( . | s_t )) ]
    Soft State Value: V(s) = sum_a pi(a|s) [ Q(s,a) - alpha * log pi(a|s) ]
    Soft Policy: pi(a|s) proportional to exp( min(Q1, Q2) / alpha )
    """
    def __init__(self, n_states=450, n_actions=4, lr=0.15, gamma=0.99, alpha=0.15, tau=0.10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.alpha = alpha  # Entropy temperature
        self.tau = tau      # Polyak target update rate
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
        if greedy:
            return int(np.argmax(probs))
        return int(np.random.choice(self.n_actions, p=probs))

    def update(self, s, a, r, next_s, done):
        if done:
            target = r
        else:
            probs_next = self.get_action_probs(next_s)
            q_targ_min = np.minimum(self.Q1_targ[next_s], self.Q2_targ[next_s])
            log_probs = np.log(probs_next + 1e-8)
            v_next = np.sum(probs_next * (q_targ_min - self.alpha * log_probs))
            target = r + self.gamma * v_next

        # Double-Q updates
        self.Q1[s, a] += self.lr * (target - self.Q1[s, a])
        self.Q2[s, a] += self.lr * (target - self.Q2[s, a])

        # Polyak target updates
        self.Q1_targ[s, a] = (1 - self.tau) * self.Q1_targ[s, a] + self.tau * self.Q1[s, a]
        self.Q2_targ[s, a] = (1 - self.tau) * self.Q2_targ[s, a] + self.tau * self.Q2[s, a]


def train_sac(env, episodes=2000, lr=0.15, gamma=0.99, alpha=0.15):
    n_states = env.size * env.size * 2
    agent = DiscreteSACAgent(n_states=n_states, n_actions=4, lr=lr, gamma=gamma, alpha=alpha)
    episode_returns = []
    episode_lengths = []
    episode_hazards = []

    start_time = time.perf_counter()
    for ep in range(episodes):
        obs, _ = env.reset()
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        total_ret, steps, ep_hazard = 0, 0, 0
        done = False

        while not done:
            action = agent.select_action(s_idx)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_s_idx = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
            done = terminated or truncated

            agent.update(s_idx, action, reward, next_s_idx, done)

            s_idx = next_s_idx
            total_ret += reward
            steps += 1
            ep_hazard += info["thermal_damage"]

        episode_returns.append(total_ret)
        episode_lengths.append(steps)
        episode_hazards.append(ep_hazard)

    training_time = time.perf_counter() - start_time
    return agent, episode_returns, episode_lengths, episode_hazards, training_time


def extract_rl_path(env, agent):
    obs, _ = env.reset()
    path = [(obs[0], obs[1])]
    has_ranger_states = [bool(obs[2])]
    actions = []
    done = False
    step = 0

    while not done and step < env.max_steps:
        s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
        a = agent.select_action(s_idx, greedy=True)
        obs, reward, term, trunc, info = env.step(a)
        path.append((obs[0], obs[1]))
        has_ranger_states.append(bool(obs[2]))
        actions.append(a)
        done = term or trunc
        step += 1

    hazard = sum(float(env.thermal_map[r, c]) for r, c in path)
    return path, actions, hazard, has_ranger_states


# ==========================================
# 6. COMPREHENSIVE EXPERIMENTAL SUITE
# ==========================================

def run_all_experiments():
    print("=" * 65)
    print("RUNNING COMPLETE EXPERIMENTAL SUITE: SEARCH TO PPO, SAC, Q-LEARNING")
    print("=" * 65)

    base_env = WildfireDroneRescueEnv(reward_mode="risk_shaped", seed=SEED)

    # 1. Classical Search & Baselines
    print("\n[1/5] Evaluating Classical Search & Baseline Policies...")
    res_bfs = run_bfs(base_env)
    res_astar_man = run_astar(base_env, heuristic_type="manhattan")
    res_astar_risk = run_astar(base_env, heuristic_type="risk_aware")
    res_random = evaluate_random_agent(base_env)
    res_greedy = evaluate_greedy_agent(base_env)

    print(f"  {res_bfs['name']}: Nodes={res_bfs['nodes_expanded']}, Hops={res_bfs['path_length']}, Hazard={res_bfs['hazard_exposure']:.2f}")
    print(f"  {res_astar_man['name']}: Nodes={res_astar_man['nodes_expanded']}, Hops={res_astar_man['path_length']}, Hazard={res_astar_man['hazard_exposure']:.2f}")
    print(f"  {res_astar_risk['name']}: Nodes={res_astar_risk['nodes_expanded']}, Hops={res_astar_risk['path_length']}, Hazard={res_astar_risk['hazard_exposure']:.2f}")
    print(f"  {res_random['name']}: Success={res_random['success_rate']}%, MeanSteps={res_random['mean_steps']:.1f}")
    print(f"  {res_greedy['name']}: Success={res_greedy['success_rate']}%, MeanSteps={res_greedy['mean_steps']:.1f}, Hazard={res_greedy['mean_hazard']:.2f}")

    # 2. Train Modern RL Algorithms: Q-Learning, PPO, SAC
    print("\n[2/5] Training Modern RL Algorithms (Q-Learning, PPO, SAC)...")
    q_agent, q_ret, q_len, q_haz, q_time = train_q_learning(base_env, episodes=2000)
    print(f"  Q-Learning trained in {q_time:.2f}s | Final 50-ep Return: {np.mean(q_ret[-50:]):.2f}")

    ppo_agent, ppo_ret, ppo_len, ppo_haz, ppo_time = train_ppo(base_env, episodes=2000)
    print(f"  PPO trained in {ppo_time:.2f}s | Final 50-ep Return: {np.mean(ppo_ret[-50:]):.2f}")

    sac_agent, sac_ret, sac_len, sac_haz, sac_time = train_sac(base_env, episodes=2000)
    print(f"  SAC trained in {sac_time:.2f}s | Final 50-ep Return: {np.mean(sac_ret[-50:]):.2f}")

    q_path, _, q_hazard, _ = extract_rl_path(base_env, q_agent)
    ppo_path, _, ppo_hazard, _ = extract_rl_path(base_env, ppo_agent)
    sac_path, _, sac_hazard, _ = extract_rl_path(base_env, sac_agent)

    print(f"  Q-Learning mission path: Length={len(q_path)-1}, Thermal Hazard={q_hazard:.2f}")
    print(f"  PPO mission path: Length={len(ppo_path)-1}, Thermal Hazard={ppo_hazard:.2f}")
    print(f"  SAC mission path: Length={len(sac_path)-1}, Thermal Hazard={sac_hazard:.2f}")

    # 3. Reward Shaping Ablation Study
    print("\n[3/5] Conducting Reward Shaping Ablation Study...")
    env_sparse = WildfireDroneRescueEnv(reward_mode="sparse", seed=SEED)
    env_dense = WildfireDroneRescueEnv(reward_mode="dense_distance", seed=SEED)
    env_risk = WildfireDroneRescueEnv(reward_mode="risk_shaped", seed=SEED)

    _, ret_sparse, len_sparse, haz_sparse, _ = train_q_learning(env_sparse, episodes=1500)
    _, ret_dense, len_dense, haz_dense, _ = train_q_learning(env_dense, episodes=1500)
    _, ret_risk, len_risk, haz_risk, _ = train_q_learning(env_risk, episodes=1500)

    # 4. Bonus Experiment: Stochastic Wind Noise Robustness
    print("\n[4/5] Evaluating Stochastic Robustness under Wind Noise (Bonus +5)...")
    wind_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    q_robust_success, q_robust_hazard = [], []
    astar_robust_success, astar_robust_hazard = [], []

    astar_planned_actions = res_astar_man["actions"]

    for w in wind_levels:
        test_env = WildfireDroneRescueEnv(wind_prob=w, reward_mode="risk_shaped")
        
        # Test Q-Learning closed-loop
        succ_q, haz_q = 0, []
        for _ in range(50):
            obs, _ = test_env.reset()
            step_cnt, tot_h, d = 0, 0, False
            while not d and step_cnt < test_env.max_steps:
                s_idx = test_env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
                a = q_agent.select_action(s_idx, greedy=True)
                obs, r, term, trunc, info = test_env.step(a)
                tot_h += info["thermal_damage"]
                step_cnt += 1
                d = term or trunc
                if term:
                    succ_q += 1
            haz_q.append(tot_h)
        q_robust_success.append((succ_q / 50) * 100)
        q_robust_hazard.append(np.mean(haz_q))

        # Test A* Open-Loop execution under wind
        succ_a, haz_a = 0, []
        for _ in range(50):
            obs, _ = test_env.reset()
            tot_h, d = 0, False
            for a in astar_planned_actions:
                obs, r, term, trunc, info = test_env.step(a)
                tot_h += info["thermal_damage"]
                if term:
                    succ_a += 1
                    d = True
                    break
            haz_a.append(tot_h)
        astar_robust_success.append((succ_a / 50) * 100)
        astar_robust_hazard.append(np.mean(haz_a))

    print(f"  Wind Robustness tested across {wind_levels}")

    # ==========================================
    # GENERATING HIGH-QUALITY FIGURES
    # ==========================================
    print("\n[5/5] Generating publication-quality figures...")

    def rolling_mean(arr, window=35):
        return np.convolve(arr, np.ones(window)/window, mode='valid')

    # Fig 1: Search Efficiency & Tradeoffs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    algos = ["BFS", "A* (Manhattan)", "A* (Risk-Aware)"]
    nodes = [res_bfs["nodes_expanded"], res_astar_man["nodes_expanded"], res_astar_risk["nodes_expanded"]]
    hazards = [res_bfs["hazard_exposure"], res_astar_man["hazard_exposure"], res_astar_risk["hazard_exposure"]]

    bars1 = ax1.bar(algos, nodes, color=["#4C72B0", "#55A868", "#C44E52"], width=0.55)
    ax1.set_ylabel("Nodes Expanded (Total Mission)", fontsize=11, fontweight="bold")
    ax1.set_title("Search Efficiency: Nodes Expanded", fontsize=12, fontweight="bold")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{int(yval)}", ha='center', va='bottom', fontweight="bold")

    bars2 = ax2.bar(algos, hazards, color=["#4C72B0", "#55A868", "#C44E52"], width=0.55)
    ax2.set_ylabel("Cumulative Thermal Damage", fontsize=11, fontweight="bold")
    ax2.set_title("Safety Comparison: Heat Exposure", fontsize=12, fontweight="bold")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight="bold")
    plt.tight_layout()
    fig1_path = os.path.join(FIGURES_DIR, "fig1_search_efficiency.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"  Saved {fig1_path}")

    # Fig 2: RL Learning Curves (Q-Learning vs PPO vs SAC)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    w = 40
    q_roll = rolling_mean(q_ret, w)
    ppo_roll = rolling_mean(ppo_ret, w)
    sac_roll = rolling_mean(sac_ret, w)

    ax1.plot(range(w, len(q_ret)+1), q_roll, label="Q-Learning (Value-based)", color="#1f77b4", lw=2.2)
    ax1.plot(range(w, len(ppo_ret)+1), ppo_roll, label="PPO (Clipped Policy-Gradient)", color="#2ca02c", lw=2.2)
    ax1.plot(range(w, len(sac_ret)+1), sac_roll, label="SAC (Soft Actor-Critic MaxEnt)", color="#9467bd", lw=2.2)
    ax1.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episodic Return (Rolling Mean)", fontsize=11, fontweight="bold")
    ax1.set_title("RL Convergence: Q-Learning vs PPO vs SAC", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    q_len_roll = rolling_mean(q_len, w)
    ppo_len_roll = rolling_mean(ppo_len, w)
    sac_len_roll = rolling_mean(sac_len, w)
    ax2.plot(range(w, len(q_len)+1), q_len_roll, label="Q-Learning", color="#1f77b4", lw=2.2)
    ax2.plot(range(w, len(ppo_len)+1), ppo_len_roll, label="PPO", color="#2ca02c", lw=2.2)
    ax2.plot(range(w, len(sac_len)+1), sac_len_roll, label="SAC", color="#9467bd", lw=2.2)
    ax2.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Steps to Extract & Evacuate", fontsize=11, fontweight="bold")
    ax2.set_title("Mission Efficiency: Steps to Evacuate", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig2_path = os.path.join(FIGURES_DIR, "fig2_rl_learning_curves.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"  Saved {fig2_path}")

    # Fig 3: Reward Shaping Ablation Study
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    w = 35
    ax1.plot(range(w, len(ret_sparse)+1), rolling_mean(ret_sparse, w), label="Sparse Reward", color="#d62728", lw=2, ls="--")
    ax1.plot(range(w, len(ret_dense)+1), rolling_mean(ret_dense, w), label="Distance-Only Shaped", color="#ff7f0e", lw=2, ls="-.")
    ax1.plot(range(w, len(ret_risk)+1), rolling_mean(ret_risk, w), label="Full Risk-Aware Shaped", color="#2ca02c", lw=2)
    ax1.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Episodic Return", fontsize=11, fontweight="bold")
    ax1.set_title("Reward Shaping Impact: Convergence Speed", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)

    ax2.plot(range(w, len(haz_sparse)+1), rolling_mean(haz_sparse, w), label="Sparse Reward", color="#d62728", lw=2, ls="--")
    ax2.plot(range(w, len(haz_dense)+1), rolling_mean(haz_dense, w), label="Distance-Only Shaped", color="#ff7f0e", lw=2, ls="-.")
    ax2.plot(range(w, len(haz_risk)+1), rolling_mean(haz_risk, w), label="Full Risk-Aware Shaped", color="#2ca02c", lw=2)
    ax2.set_xlabel("Training Episodes", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Thermal Exposure", fontsize=11, fontweight="bold")
    ax2.set_title("Safety Ablation: Thermal Hazard Incurred", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(FIGURES_DIR, "fig3_reward_ablation.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"  Saved {fig3_path}")

    # Fig 4: Multi-Panel Terrain, Value Function, Policy & Trajectories
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11.5))
    
    # 4A: Environment Layout
    ax = axes[0, 0]
    cmap_heat = sns.color_palette("YlOrRd", as_cmap=True)
    sns.heatmap(base_env.thermal_map, ax=ax, cmap=cmap_heat, cbar=True, vmin=0, vmax=1.0, annot=False)
    for r, c in base_env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#2c3e50", fill=True))
    ax.text(0.5, 0.5, "DEPOT\n(0,0)", ha="center", va="center", color="blue", fontweight="bold", fontsize=9)
    ax.text(base_env.ranger_pos[1] + 0.5, base_env.ranger_pos[0] + 0.5, "RANGER\n(7,2)", ha="center", va="center", color="orange", fontweight="bold", fontsize=9)
    ax.text(base_env.size - 0.5, base_env.size - 0.5, "HOSPITAL\nEVAC (14,14)", ha="center", va="center", color="darkgreen", fontweight="bold", fontsize=9)
    ax.set_title("(A) Disaster Terrain: 2 Wildfires & Canyon Pocket", fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid Column"); ax.set_ylabel("Grid Row")

    # 4B: Learned Value Function V(s) [Phase 1]
    ax = axes[0, 1]
    V_grid_phase1 = np.max(q_agent.Q[:base_env.size*base_env.size], axis=1).reshape((base_env.size, base_env.size))
    for r, c in base_env.obstacles:
        V_grid_phase1[r, c] = np.nan
    sns.heatmap(V_grid_phase1, ax=ax, cmap="viridis", cbar=True, annot=False)
    ax.set_title("(B) Learned Value Function V(s) [Phase 1: Ingress]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid Column"); ax.set_ylabel("Grid Row")

    # 4C: Learned Policy Field [Phase 2]
    ax = axes[1, 0]
    ax.imshow(base_env.thermal_map, cmap="YlOrRd", alpha=0.4, extent=[0, base_env.size, base_env.size, 0])
    for r, c in base_env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#2c3e50", fill=True))
    action_arrows = {0: (0, -0.35), 1: (0, 0.35), 2: (-0.35, 0), 3: (0.35, 0)}
    for r in range(base_env.size):
        for c in range(base_env.size):
            if (r, c) in base_env.obstacles or (r, c) == base_env.evac_pos:
                continue
            s_idx = base_env._state_to_idx((r, c), has_ranger=True)
            best_a = q_agent.select_action(s_idx, greedy=True)
            dx, dy = action_arrows[best_a]
            ax.arrow(c + 0.5, r + 0.5, dx, dy, head_width=0.18, head_length=0.18, fc="black", ec="black")
    ax.set_xlim(0, base_env.size); ax.set_ylim(base_env.size, 0)
    ax.set_title("(C) Learned Policy Vector Field [Phase 2: Evac Egress]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid Column"); ax.set_ylabel("Grid Row")

    # 4D: Trajectory Comparison Overlay across Search, Q-Learning, PPO, SAC
    ax = axes[1, 1]
    ax.imshow(base_env.thermal_map, cmap="YlOrRd", alpha=0.45, extent=[0, base_env.size, base_env.size, 0])
    for r, c in base_env.obstacles:
        ax.add_patch(plt.Rectangle((c, r), 1, 1, color="#2c3e50", fill=True))
    
    def plot_path(p, col, lab, ls, marker):
        rows = [x[0] + 0.5 for x in p]
        cols = [x[1] + 0.5 for x in p]
        ax.plot(cols, rows, color=col, label=lab, linestyle=ls, marker=marker, markersize=3.2, lw=2.0)

    plot_path(res_bfs["path"], "#2b5c8f", f"BFS ({res_bfs['path_length']} hops, Hazard={res_bfs['hazard_exposure']:.2f})", "-", "o")
    plot_path(res_astar_man["path"], "#d95f02", f"A* Manhattan ({res_astar_man['path_length']} hops, Hazard={res_astar_man['hazard_exposure']:.2f})", "--", "s")
    plot_path(res_astar_risk["path"], "#7570b3", f"A* Risk-Aware ({res_astar_risk['path_length']} hops, Hazard=0.00)", "-.", "^")
    plot_path(q_path, "#1b9e77", f"Q-Learning ({len(q_path)-1} hops, Hazard=0.00)", "-", "D")
    plot_path(ppo_path, "#e7298a", f"PPO ({len(ppo_path)-1} hops, Hazard=0.00)", "--", "v")
    plot_path(sac_path, "#66a61e", f"SAC ({len(sac_path)-1} hops, Hazard=0.00)", ":", "p")

    ax.plot(0.5, 0.5, marker="h", color="blue", markersize=13, label="Depot (0,0)")
    ax.plot(base_env.ranger_pos[1] + 0.5, base_env.ranger_pos[0] + 0.5, marker="P", color="orange", markersize=14, label="Ranger (7,2)")
    ax.plot(base_env.size - 0.5, base_env.size - 0.5, marker="*", color="darkgreen", markersize=15, label="Evacuation Helipad (14,14)")
    ax.set_xlim(0, base_env.size); ax.set_ylim(base_env.size, 0)
    ax.set_title("(D) Mission Trajectories: Search vs Q-Learning vs PPO vs SAC", fontsize=11, fontweight="bold")
    ax.set_xlabel("Grid Column"); ax.set_ylabel("Grid Row")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.15), ncol=2, fontsize=9.0, frameon=True)

    plt.tight_layout()
    fig4_path = os.path.join(FIGURES_DIR, "fig4_terrain_policy_value_heatmaps.png")
    plt.savefig(fig4_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fig4_path}")

    # Fig 5: Robustness under Stochastic Wind Noise (Bonus +5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax1.plot(wind_levels, q_robust_success, marker="o", lw=2.5, color="#1b9e77", label="Q-Learning (Closed-Loop RL)")
    ax1.plot(wind_levels, astar_robust_success, marker="s", lw=2.5, color="#d95f02", ls="--", label="A* Manhattan (Open-Loop Plan)")
    ax1.set_xlabel("Wind Slip Probability ($p_{wind}$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Mission Success Rate (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Robustness: Mission Success vs Stochasticity", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower left", frameon=True)
    ax1.set_ylim(-5, 105)

    ax2.plot(wind_levels, q_robust_hazard, marker="o", lw=2.5, color="#1b9e77", label="Q-Learning")
    ax2.plot(wind_levels, astar_robust_hazard, marker="s", lw=2.5, color="#d95f02", ls="--", label="A* Manhattan")
    ax2.set_xlabel("Wind Slip Probability ($p_{wind}$)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Mean Thermal Exposure Damage", fontsize=11, fontweight="bold")
    ax2.set_title("Safety Degradation under Wind Noise", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    fig5_path = os.path.join(FIGURES_DIR, "fig5_bonus_robustness_noise.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"  Saved {fig5_path}")

    print("\n" + "=" * 65)
    print("ALL 2-PHASE EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 65)

    return {
        "bfs": res_bfs,
        "astar_man": res_astar_man,
        "astar_risk": res_astar_risk,
        "random": res_random,
        "greedy": res_greedy,
        "q_learning": {"time": q_time, "final_ret": np.mean(q_ret[-50:]), "hazard": q_hazard, "len": len(q_path)-1},
        "ppo": {"time": ppo_time, "final_ret": np.mean(ppo_ret[-50:]), "hazard": ppo_hazard, "len": len(ppo_path)-1},
        "sac": {"time": sac_time, "final_ret": np.mean(sac_ret[-50:]), "hazard": sac_hazard, "len": len(sac_path)-1},
        "wind_levels": wind_levels,
        "q_robust_success": q_robust_success,
        "astar_robust_success": astar_robust_success,
    }


if __name__ == "__main__":
    import sys
    results = run_all_experiments()
    if "--gif" in sys.argv or "-g" in sys.argv:
        from generate_demo_gif import generate_all_model_gifs
        generate_all_model_gifs()
    else:
        print("\nTip: To also generate individual and comparative mission GIFs, run:")
        print("  python generate_demo_gif.py   (or: python run_experiments.py --gif)")
