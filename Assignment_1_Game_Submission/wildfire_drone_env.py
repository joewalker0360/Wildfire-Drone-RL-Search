"""
WildfireDroneRescueEnv: Custom Gymnasium Environment (Two-Phase Search, Rescue & Evacuation Mission)
Course: Machine Learning (23MEE305) - Assignment 1
Scenario:
  Phase 1: Ingress & Search from Launch Depot (0,0) to Trapped Ranger in Middle Canyon Pocket (7,2).
  Phase 2: Egress & Evacuation from Canyon Pocket (7,2) to Hospital Evacuation Base (14,14).
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class WildfireDroneRescueEnv(gym.Env):
    """
    Intricate 15x15 Two-Phase Disaster Search & Rescue Mission.
    
    Mission Objectives:
      1. Launch from Base Depot (0, 0).
      2. Locate and extract the trapped ranger at interior rock pocket (7, 2).
      3. Evacuate the ranger safely to the Hospital Evacuation Base at (14, 14).
      
    Hazard Layout:
      - Fire Zone 1 (Entrance Gorge): Rows 3-5, Col 5 (blocks direct ingress).
      - Fire Zone 2 (Evac Ridge Blaze): Rows 9-12, Col 8 (blocks direct egress to hospital).
      - Safe bypass corridors on west and east allowing zero-damage extraction.
    """
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
        self.ranger_pos = (7, 2)    # Ranger trapped in secluded canyon pocket
        self.evac_pos = (size - 1, size - 1)  # Emergency Evacuation Base

        # Build Mountain Barriers
        self.obstacles = set()

        # 1. Perimeter Mountain Cliffs
        for r in range(2, 13): self.obstacles.add((r, 0))   # West perimeter cliff
        for c in range(2, 13): self.obstacles.add((14, c))  # South perimeter cliff
        for c in range(2, 13): self.obstacles.add((0, c))   # North perimeter cliff
        for r in range(2, 13): self.obstacles.add((r, 14))  # East perimeter cliff

        # 2. Canyon Pocket Enclosing Ranger at (7, 2) on 3 sides
        self.obstacles.add((6, 1)); self.obstacles.add((6, 2)); self.obstacles.add((6, 3))
        self.obstacles.add((8, 1)); self.obstacles.add((8, 2)); self.obstacles.add((8, 3))
        self.obstacles.add((7, 1))  # Closed on left, top, bottom; opening ONLY at (7, 3)!

        # 3. Mountain Ridge 1 (Row 4): Openings at Col 1 (West Detour), Col 5 (Direct/Fire 1), Col 13 (East)
        for c in range(1, 14):
            if c not in (1, 5, 13):
                self.obstacles.add((4, c))

        # 4. Mountain Ridge 2 (Row 10): Openings at Col 3, Col 8 (Direct/Fire 2), Col 13 (East)
        for c in range(1, 14):
            if c not in (3, 8, 13):
                self.obstacles.add((10, c))

        # 5. East Alpine Weaving Baffles
        for c in range(11, 14):
            self.obstacles.add((7, c))
            self.obstacles.add((12, c))

        # -------------------------------------------------------------
        # TWO STRATEGIC WILDFIRE HAZARD ZONES
        # -------------------------------------------------------------
        self.thermal_map = np.zeros((size, size), dtype=np.float32)

        # FIRE ZONE 1: Entrance Wildfire Gorge (Rows 3-5, Col 5)
        for r in range(3, 6):
            self.thermal_map[r, 5] = 1.0   # Core flame
            self.thermal_map[r, 6] = 0.5   # Smoke East
        for r in range(3, 5):
            self.thermal_map[r, 4] = 0.5   # Smoke West (leaving (5, 4) clear for safe canyon entry)

        # FIRE ZONE 2: Evacuation Corridor Wildfire (Rows 9-12, Col 8)
        for r in range(9, 13):
            self.thermal_map[r, 8] = 1.0   # Core flame
            self.thermal_map[r, 7] = 0.5   # Smoke
            self.thermal_map[r, 9] = 0.5   # Smoke

        self.action_effects = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, -1),  # LEFT
            3: (0, 1),   # RIGHT
        }

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

        # Wind slip dynamics
        effective_action = action
        if self.wind_prob > 0.0 and self.np_random.random() < self.wind_prob:
            effective_action = self.np_random.choice([1, 3])

        dr, dc = self.action_effects[effective_action]
        nr = prev_state[0] + dr
        nc = prev_state[1] + dc

        if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
            self.state = (nr, nc)
        else:
            self.state = prev_state

        next_state = self.state

        # Check pickup event
        just_picked_up = False
        if not self.has_ranger and next_state == self.ranger_pos:
            self.has_ranger = True
            just_picked_up = True

        # Check final mission termination (delivered to evacuation base)
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
        else:
            raise ValueError(f"Unknown reward mode: {self.reward_mode}")

        info = {
            "thermal_damage": thermal_damage,
            "has_ranger": self.has_ranger,
            "reached_evac": terminated,
            "steps": self.current_step,
        }

        obs = np.array([self.state[0], self.state[1], int(self.has_ranger)], dtype=np.int32)
        return obs, reward, terminated, truncated, info

    def get_valid_actions(self, state):
        valid = []
        for a, (dr, dc) in self.action_effects.items():
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                valid.append(a)
        return valid

    def get_neighbors(self, state):
        neighbors = []
        for a, (dr, dc) in self.action_effects.items():
            nr, nc = state[0] + dr, state[1] + dc
            if 0 <= nr < self.size and 0 <= nc < self.size and (nr, nc) not in self.obstacles:
                thermal = float(self.thermal_map[nr, nc])
                neighbors.append(((nr, nc), a, thermal))
        return neighbors
