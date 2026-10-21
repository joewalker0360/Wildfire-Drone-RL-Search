"""
interactive_learning.py
Cross-Maze Generalized & Interactive Reinforcement Learning with Human Feedback (RLHF),
Human Demonstrations, and Real-time Policy Interventions for the Wildfire Drone Rescue System.
"""

import numpy as np
from collections import defaultdict


class InteractiveQLearningAgent:
    """
    Enhanced Dual-System Q-Learning agent designed for Continuous Learning across Randomized Mazes:
    1. Relational Sensory Q-Table (Q_rel): Generalizes navigation, hazard dodging, and obstacle
       avoidance across arbitrary procedural mazes. Persists across all successful runs!
    2. Local Episodic Q-Table (Q_local): Adapts rapidly to specific bottlenecks of the active maze.
    3. Real-Time Human Intervention Learning (DAgger): Updates both tables when user steers.
    4. Real-Time Evaluative Feedback (TAMER / RLHF): Modifies action-values based on user praise/penalties.
    5. Batch Imitation from User Demonstrations: Replays user flights with human guidance bonuses.
    """

    def __init__(
        self,
        n_states=450,
        n_actions=4,
        lr=0.20,
        gamma=0.99,
        eps_start=0.15,
        eps_min=0.01,
        eps_decay=0.995,
        human_feedback_weight=35.0,
        base_q_table=None
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_start
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.human_feedback_weight = human_feedback_weight

        # Local coordinate Q-table (450 states x 4 actions)
        if base_q_table is not None:
            self.Q_local = np.copy(base_q_table)
        else:
            self.Q_local = np.zeros((n_states, n_actions), dtype=np.float32)

        # Cross-Maze Relational Q-Table: default dictionary mapping relational tuples to 4 action values
        # Relational Key: (rel_dir_r, rel_dir_c, has_ranger, walls_tuple, hazards_tuple)
        self.Q_rel = defaultdict(lambda: np.zeros(self.n_actions, dtype=np.float32))

        # Initialize sensible relational priors (seek target, avoid direct flame/walls)
        self._init_relational_priors()

        # Telemetry & Learning stats
        self.human_interventions_count = 0
        self.human_feedbacks_count = 0
        self.demonstrations_learned_count = 0
        self.mazes_cleared_count = 0
        self.total_online_updates = 0

    def _init_relational_priors(self):
        """Initializes relational navigation tendencies (instincts)."""
        action_effects = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                for ranger in [0, 1]:
                    for w in [(0,0,0,0)]:
                        for h in [(0,0,0,0)]:
                            key = (dr, dc, ranger, w, h)
                            q_arr = np.zeros(self.n_actions, dtype=np.float32)
                            for a, (adr, adc) in action_effects.items():
                                alignment = (adr * dr) + (adc * dc)
                                q_arr[a] += alignment * 8.0
                            self.Q_rel[key] = q_arr

    def get_relational_key(self, env):
        """Extracts the relational feature key from the environment."""
        obs = env.get_relational_observation()
        return (
            obs["rel_dir"][0],
            obs["rel_dir"][1],
            obs["has_ranger"],
            obs["walls"],
            obs["hazards"]
        )

    def get_combined_q_values(self, env, state_idx=None):
        """Combines Relational Generalization Q-values with Local Coordinate Q-values."""
        if state_idx is None:
            state_idx = env._state_to_idx(env.state, env.has_ranger)

        rel_key = self.get_relational_key(env)
        q_rel = self.Q_rel[rel_key]
        q_loc = self.Q_local[state_idx]

        # Relational weights carry cross-maze intuition; local values refine local coordinates
        return q_rel * 1.2 + q_loc * 0.8

    def select_action(self, env, state_idx=None, greedy=False):
        """Selects action using combined cross-maze and local Q-values."""
        if not greedy and np.random.random() < self.epsilon:
            return int(np.random.randint(self.n_actions))

        q_vals = self.get_combined_q_values(env, state_idx)
        max_val = np.max(q_vals)
        best_actions = np.where(q_vals == max_val)[0]
        return int(np.random.choice(best_actions))

    def update(self, env, s, a, r, next_s, done, lr_override=None):
        """Dual-system Bellman update on both local and relational cross-maze tables."""
        alpha = lr_override if lr_override is not None else self.lr

        # 1. Local Coordinate Update
        td_target_loc = r if done else r + self.gamma * np.max(self.Q_local[next_s])
        td_error_loc = td_target_loc - self.Q_local[s, a]
        self.Q_local[s, a] += alpha * td_error_loc

        # 2. Relational Cross-Maze Update
        rel_key = self.get_relational_key(env)
        max_next_rel = 0.0 if done else np.max(self.Q_rel[rel_key])
        td_target_rel = r if done else r + self.gamma * max_next_rel
        td_error_rel = td_target_rel - self.Q_rel[rel_key][a]
        self.Q_rel[rel_key][a] += alpha * td_error_rel

        self.total_online_updates += 1
        return float(td_error_loc)

    def learn_from_intervention(self, env, s, human_action, r, next_s, done):
        """
        Updates agent when human pilot overrides the autonomous decision.
        Teaches both the current coordinate and the universal relational condition!
        """
        guidance_bonus = 25.0
        effective_r = r + guidance_bonus
        td_err = self.update(env, s, human_action, effective_r, next_s, done, lr_override=0.35)
        self.human_interventions_count += 1
        return td_err

    def apply_human_feedback(self, env, s, a, valence):
        """
        TAMER / RLHF: Instant feedback (+1 Praise / -1 Scold) applied to both tables.
        """
        feedback_delta = self.human_feedback_weight * valence
        self.Q_local[s, a] += feedback_delta

        rel_key = self.get_relational_key(env)
        self.Q_rel[rel_key][a] += feedback_delta * 1.5

        self.human_feedbacks_count += 1
        return float(self.Q_local[s, a])

    def learn_from_demonstration(self, env, trajectory, epochs=8, lr=0.30):
        """Replays an entire human flight across both local and cross-maze representations."""
        if not trajectory:
            return 0

        total_updates = 0
        for _ in range(epochs):
            for step in reversed(trajectory):
                s = step["s_idx"]
                a = step["action"]
                r = step["reward"] + 15.0
                next_s = step["next_s_idx"]
                done = step["done"]

                self.update(env, s, a, r, next_s, done, lr_override=lr)
                total_updates += 1

        self.demonstrations_learned_count += 1
        return total_updates

    def on_maze_cleared(self):
        """
        Called when the drone successfully clears a maze/level.
        Preserves cross-maze relational intuition (Q_rel) while softly resetting
        local coordinates so the agent can quickly master the new layout!
        """
        self.mazes_cleared_count += 1
        # Soft decay on local coordinate table so new maze coordinates don't clash
        self.Q_local *= 0.15
        # Exploration rate slightly boosted to discover novel corridors in new maze
        self.epsilon = min(0.18, self.epsilon + 0.05)

    def quick_fine_tune(self, env, episodes=50, progress_callback=None):
        """Fast episodic fine-tune on current randomized maze."""
        returns = []
        for ep in range(episodes):
            obs, _ = env.reset()
            s_idx = env._state_to_idx((obs[0], obs[1]), bool(obs[2]))
            tot_ret, done = 0.0, False

            while not done:
                action = self.select_action(env, s_idx, greedy=False)
                next_obs, reward, term, trunc, info = env.step(action)
                next_s_idx = env._state_to_idx((next_obs[0], next_obs[1]), bool(next_obs[2]))
                done = term or trunc

                self.update(env, s_idx, action, reward, next_s_idx, done)
                s_idx = next_s_idx
                tot_ret += reward

            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
            returns.append(tot_ret)
            if progress_callback:
                progress_callback(ep + 1, episodes, tot_ret)

        return returns

    def get_policy_and_values(self, env, has_ranger=False):
        """Extracts 2D grid of policy direction arrows and state values for visualization."""
        action_arrows = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        policy_grid = np.empty((env.size, env.size), dtype=object)
        value_grid = np.zeros((env.size, env.size), dtype=np.float32)

        # Save current state to restore after probe
        original_state = env.state
        original_has_ranger = env.has_ranger

        for r in range(env.size):
            for c in range(env.size):
                if (r, c) in env.obstacles:
                    policy_grid[r, c] = "■"
                    value_grid[r, c] = -999.0
                    continue

                env.state = (r, c)
                env.has_ranger = has_ranger
                s_idx = env._state_to_idx((r, c), has_ranger)
                q_vals = self.get_combined_q_values(env, s_idx)
                best_a = int(np.argmax(q_vals))
                best_val = float(np.max(q_vals))

                policy_grid[r, c] = action_arrows[best_a]
                value_grid[r, c] = best_val

        # Restore state
        env.state = original_state
        env.has_ranger = original_has_ranger

        return policy_grid, value_grid
