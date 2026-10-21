"""
dynamic_wildfire_env.py
Extended dynamic and randomized environment for the Wildfire Drone Rescue mission.
Supports:
1. Procedural Maze & Obstacle Layout Generation with guaranteed BFS reachability.
2. Procedural Wildfire Clusters with dynamic core and smoke halos.
3. Automatic Maze Randomization on Successful Mission Delivery (Level Progression).
4. Dynamic fire spreading over time.
5. Water drop / fire suppression mechanics.
6. Relational sensory observations for cross-maze generalization in RL.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from wildfire_drone_env import WildfireDroneRescueEnv


class DynamicWildfireDroneEnv(WildfireDroneRescueEnv):
    """
    Dynamic and procedurally randomized variant of the Wildfire Drone Rescue Environment.
    Features:
    - Procedural maze generation with guaranteed reachability.
    - Automatic maze/scenario re-randomization upon each successful mission.
    - Dynamic difficulty scaling across successful run streaks.
    """

    def __init__(
        self,
        size=15,
        wind_prob=0.0,
        reward_mode="risk_shaped",
        max_steps=220,
        fire_spread_prob=0.0,
        water_capacity=3,
        auto_randomize_on_success=True,
        seed=None
    ):
        super().__init__(size=size, wind_prob=wind_prob, reward_mode=reward_mode, max_steps=max_steps, seed=seed)
        self.fire_spread_prob = fire_spread_prob
        self.water_capacity = water_capacity
        self.water_remaining = water_capacity
        self.fires_extinguished = 0
        self.auto_randomize_on_success = auto_randomize_on_success

        # Level progression & success streak
        self.successful_runs_count = 0
        self.current_level = 1
        self.level_benchmark_steps = 40  # Set by RL model after scouting
        self.step_allowance = 2  # 2-step allowance for human player
        self.level_failed = False

        # Snapshots for restarting the current level upon failure
        self.saved_obstacles = set(self.obstacles)
        self.saved_start_pos = self.start_pos
        self.saved_ranger_pos = self.ranger_pos
        self.saved_evac_pos = self.evac_pos
        self.saved_thermal_map = np.copy(self.thermal_map)

        # Cache canonical layout for reset
        self.default_obstacles = set(self.obstacles)
        self.default_start_pos = self.start_pos
        self.default_ranger_pos = self.ranger_pos
        self.default_evac_pos = self.evac_pos
        self.default_thermal_map = np.copy(self.thermal_map)

    def save_level_snapshot(self):
        """Saves current level layout so the human can retry the same level if failed."""
        self.saved_obstacles = set(self.obstacles)
        self.saved_start_pos = self.start_pos
        self.saved_ranger_pos = self.ranger_pos
        self.saved_evac_pos = self.evac_pos
        self.saved_thermal_map = np.copy(self.thermal_map)

    def restart_current_level(self):
        """Restarts the EXACT same level layout after failure."""
        self.obstacles = set(self.saved_obstacles)
        self.start_pos = self.saved_start_pos
        self.ranger_pos = self.saved_ranger_pos
        self.evac_pos = self.saved_evac_pos
        self.thermal_map = np.copy(self.saved_thermal_map)
        self.water_remaining = self.water_capacity
        self.fires_extinguished = 0
        self.max_steps = self.level_benchmark_steps + self.step_allowance
        self.level_failed = False
        return self.reset()

    def is_traversable(self, pos):
        """Checks if a position is within bounds and not an obstacle."""
        r, c = pos
        return 0 <= r < self.size and 0 <= c < self.size and (r, c) not in self.obstacles

    def check_solvability(self, start, ranger, evac, thermal_limit=None):
        """
        Uses BFS to check if a valid path exists from start -> ranger -> evac.
        Optionally checks if a corridor below thermal_limit exists.
        """
        def bfs_check(s, target):
            queue = deque([s])
            visited = {s}
            while queue:
                curr = queue.popleft()
                if curr == target:
                    return True
                for neighbor, _, thermal in self.get_neighbors(curr):
                    if thermal_limit is not None and thermal > thermal_limit:
                        continue
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            return False

        can_reach_ranger = bfs_check(start, ranger)
        can_reach_evac = bfs_check(ranger, evac)
        return can_reach_ranger and can_reach_evac

    def reset_to_default_layout(self):
        """Restores the canonical benchmark layout."""
        self.start_pos = self.default_start_pos
        self.ranger_pos = self.default_ranger_pos
        self.evac_pos = self.default_evac_pos
        self.obstacles = set(self.default_obstacles)
        self.thermal_map = np.copy(self.default_thermal_map)
        self.water_remaining = self.water_capacity
        self.fires_extinguished = 0
        self.current_level = 1
        return self.reset()

    def generate_procedural_maze(self, seed=None):
        """
        Generates a new procedural canyon & ridge obstacle layout.
        Maintains mountain cliff borders, interior ridges with openings,
        a canyon pocket enclosure, and boulder clusters.
        """
        rng = np.random.default_rng(seed)
        new_obs = set()

        # 1. Outer perimeter cliffs with strategic corner openings
        for r in range(2, 13):
            new_obs.add((r, 0))
            new_obs.add((r, self.size - 1))
        for c in range(2, 13):
            new_obs.add((0, c))
            new_obs.add((self.size - 1, c))

        # 2. Interior Mountain Ridge 1 (around rows 3-5)
        r1 = rng.integers(3, 6)
        openings_r1 = set(rng.choice(range(1, self.size - 1), size=3, replace=False))
        for c in range(1, self.size - 1):
            if c not in openings_r1:
                new_obs.add((r1, c))

        # 2. Interior Mountain Ridge 1 (around rows 3-5)
        r1 = int(rng.integers(3, 6))
        openings_r1 = set(int(x) for x in rng.choice(range(1, self.size - 1), size=3, replace=False))
        for c in range(1, self.size - 1):
            if c not in openings_r1:
                new_obs.add((r1, c))

        # 3. Interior Mountain Ridge 2 (around rows 9-11)
        r2 = int(rng.integers(9, 12))
        openings_r2 = set(int(x) for x in rng.choice(range(1, self.size - 1), size=3, replace=False))
        for c in range(1, self.size - 1):
            if c not in openings_r2:
                new_obs.add((r2, c))

        # 4. Interior Canyon Pocket for Trapped Ranger
        pocket_r = int(rng.integers(6, 9))
        pocket_c = int(rng.integers(2, 5))
        # 3-sided enclosure
        new_obs.add((pocket_r - 1, pocket_c - 1))
        new_obs.add((pocket_r - 1, pocket_c))
        new_obs.add((pocket_r - 1, pocket_c + 1))
        new_obs.add((pocket_r + 1, pocket_c - 1))
        new_obs.add((pocket_r + 1, pocket_c))
        new_obs.add((pocket_r + 1, pocket_c + 1))
        new_obs.add((pocket_r, pocket_c - 1))  # Open only on right at (pocket_r, pocket_c + 1)

        # 5. Scattered boulders (4-8 random rocks)
        num_boulders = int(rng.integers(4, 9))
        for _ in range(num_boulders):
            br = int(rng.integers(1, self.size - 1))
            bc = int(rng.integers(1, self.size - 1))
            if (br, bc) != (pocket_r, pocket_c):
                new_obs.add((br, bc))

        return {(int(r), int(c)) for r, c in new_obs}, (pocket_r, pocket_c)

    def randomize_scenario(
        self,
        randomize_maze=True,
        randomize_agents=True,
        randomize_fires=True,
        num_fire_clusters=2,
        fire_radius=1,
        seed=None
    ):
        """
        Randomizes the entire scenario:
        - Procedural maze / obstacle layout (optional).
        - Relocates start depot, ranger, and hospital evac bases.
        - Generates procedural wildfire hazard clusters.
        - Guarantees 100% BFS solvability.
        """
        rng = np.random.default_rng(seed)

        for attempt in range(80):
            if randomize_maze:
                cand_obs, suggested_ranger = self.generate_procedural_maze(seed=None)
                self.obstacles = {(int(r), int(c)) for r, c in cand_obs}
            else:
                suggested_ranger = None

            # Free cells not blocked by obstacles
            free_cells = [
                (int(r), int(c)) for r in range(self.size) for c in range(self.size)
                if (r, c) not in self.obstacles
            ]

            if len(free_cells) < 15:
                continue

            # Randomize agent positions
            if randomize_agents:
                if suggested_ranger and suggested_ranger in free_cells and rng.random() < 0.65:
                    new_ranger = (int(suggested_ranger[0]), int(suggested_ranger[1]))
                    other_candidates = [c for c in free_cells if c != new_ranger]
                    chosen = rng.choice(len(other_candidates), size=2, replace=False)
                    new_start = other_candidates[chosen[0]]
                    new_evac = other_candidates[chosen[1]]
                else:
                    chosen = rng.choice(len(free_cells), size=3, replace=False)
                    new_start = free_cells[chosen[0]]
                    new_ranger = free_cells[chosen[1]]
                    new_evac = free_cells[chosen[2]]

                d1 = abs(new_start[0] - new_ranger[0]) + abs(new_start[1] - new_ranger[1])
                d2 = abs(new_ranger[0] - new_evac[0]) + abs(new_ranger[1] - new_evac[1])
                if d1 < 4 or d2 < 4:
                    continue

                self.start_pos = (int(new_start[0]), int(new_start[1]))
                self.ranger_pos = (int(new_ranger[0]), int(new_ranger[1]))
                self.evac_pos = (int(new_evac[0]), int(new_evac[1]))

            # Randomize fires
            if randomize_fires:
                new_thermal = np.zeros((self.size, self.size), dtype=np.float32)
                protected = {self.start_pos, self.ranger_pos, self.evac_pos}
                for pr, pc in list(protected):
                    for dr in range(-1, 2):
                        for dc in range(-1, 2):
                            protected.add((pr + dr, pc + dc))

                fire_candidates = [pos for pos in free_cells if pos not in protected]
                n_clusters = min(num_fire_clusters, len(fire_candidates))
                if n_clusters > 0:
                    centers = rng.choice(len(fire_candidates), size=n_clusters, replace=False)
                    for idx in centers:
                        cr, cc = fire_candidates[idx]
                        new_thermal[cr, cc] = 1.0
                        for dr in range(-fire_radius, fire_radius + 1):
                            for dc in range(-fire_radius, fire_radius + 1):
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                                    if (nr, nc) not in protected:
                                        dist = max(abs(dr), abs(dc))
                                        if dist == 1 and new_thermal[nr, nc] < 0.5:
                                            new_thermal[nr, nc] = 0.5
                                        elif dist == 2 and new_thermal[nr, nc] < 0.25:
                                            new_thermal[nr, nc] = 0.25

                self.thermal_map = new_thermal

            # Validate solvability
            if self.check_solvability(self.start_pos, self.ranger_pos, self.evac_pos):
                break
        else:
            # Fallback to default layout if all 80 procedural attempts fail reachability
            self.reset_to_default_layout()

        self.water_remaining = self.water_capacity
        self.fires_extinguished = 0
        return self.reset()

    def get_relational_observation(self):
        """
        Extracts goal-relative and sensory observations for cross-maze RL generalization:
        - rel_r: sign of (target_r - curr_r) (-1, 0, 1)
        - rel_c: sign of (target_c - curr_c) (-1, 0, 1)
        - has_ranger: 0 or 1
        - dist_bin: 0 (<2), 1 (2-4), 2 (5-8), 3 (>8)
        - walls: 4-tuple of booleans (up, down, left, right)
        - hazards: 4-tuple of booleans (up, down, left, right has heat > 0)
        """
        r, c = self.state
        target = self.evac_pos if self.has_ranger else self.ranger_pos
        dr = target[0] - r
        dc = target[1] - c

        sign_r = 1 if dr > 0 else (-1 if dr < 0 else 0)
        sign_c = 1 if dc > 0 else (-1 if dc < 0 else 0)
        manhattan = abs(dr) + abs(dc)
        dist_bin = 0 if manhattan <= 2 else (1 if manhattan <= 4 else (2 if manhattan <= 8 else 3))

        walls = []
        hazards = []
        for dr_s, dc_s in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr_s, c + dc_s
            is_wall = not (0 <= nr < self.size and 0 <= nc < self.size) or ((nr, nc) in self.obstacles)
            walls.append(int(is_wall))
            is_hazard = 0
            if 0 <= nr < self.size and 0 <= nc < self.size:
                is_hazard = int(self.thermal_map[nr, nc] > 0.0)
            hazards.append(is_hazard)

        return {
            "rel_dir": (sign_r, sign_c),
            "has_ranger": int(self.has_ranger),
            "dist_bin": dist_bin,
            "walls": tuple(walls),
            "hazards": tuple(hazards),
        }

    def spread_wildfire(self):
        """Simulates dynamic fire spread into adjacent traversable cells."""
        if self.fire_spread_prob <= 0.0:
            return

        new_map = np.copy(self.thermal_map)
        protected = {self.start_pos, self.ranger_pos, self.evac_pos, self.state}

        for r in range(self.size):
            for c in range(self.size):
                if self.thermal_map[r, c] >= 0.8:
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.size and 0 <= nc < self.size:
                            if (nr, nc) not in self.obstacles and (nr, nc) not in protected:
                                if self.thermal_map[nr, nc] < 0.5 and np.random.random() < self.fire_spread_prob:
                                    new_map[nr, nc] = min(1.0, new_map[nr, nc] + 0.5)

        self.thermal_map = new_map

    def extinguish_current_cell(self):
        """Drops water suppressant on the current drone position."""
        if self.water_remaining <= 0:
            return False, 0.0

        r, c = self.state
        heat = self.thermal_map[r, c]
        if heat > 0.0:
            self.thermal_map[r, c] = 0.0
            self.water_remaining -= 1
            self.fires_extinguished += 1
            suppression_reward = 35.0 * heat
            return True, suppression_reward
        return False, 0.0

    def step(self, action, human_feedback=0.0):
        """
        Step function with:
        - Auto-randomization on successful mission delivery
        - Dynamic fire spreading
        - Human feedback incorporation
        """
        obs, reward, terminated, truncated, info = super().step(action)

        if human_feedback != 0.0:
            feedback_bonus = 25.0 * human_feedback
            reward += feedback_bonus
            info["human_feedback"] = human_feedback
            info["feedback_bonus"] = feedback_bonus

        if not terminated and self.fire_spread_prob > 0.0 and self.current_step % 3 == 0:
            self.spread_wildfire()

        info["water_remaining"] = self.water_remaining
        info["fires_extinguished"] = self.fires_extinguished
        info["start_pos"] = self.start_pos
        info["ranger_pos"] = self.ranger_pos
        info["evac_pos"] = self.evac_pos
        info["current_level"] = self.current_level
        info["successful_runs"] = self.successful_runs_count
        info["relational_obs"] = self.get_relational_observation()

        # Handle successful run completion (Passed level within steps)
        if terminated and info.get("reached_evac", False):
            self.successful_runs_count += 1
            self.current_level += 1
            info["level_cleared"] = True
            info["new_level"] = self.current_level
            info["total_successes"] = self.successful_runs_count
            self.level_failed = False
        elif truncated and not terminated:
            # Failed level because steps exceeded the RL model's benchmark + 2 allowance!
            self.level_failed = True
            info["level_failed"] = True
            info["fail_reason"] = "Step allowance exceeded! The AI benchmark remains secret until you complete the level. Restarting this level from the beginning."

        return obs, reward, terminated, truncated, info
