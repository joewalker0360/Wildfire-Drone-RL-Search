# Presentation Slides: Evolution from Search to Reinforcement Learning
**Course:** Machine Learning (23MEE305) — Assignment 1  
**Project:** Two-Phase Autonomous Wildfire Search, Rescue & Evacuation Drone  
**Deliverables:** Code, Visualizations, Individual & Comparison Animated Telemetry GIFs, 4-Page Report, Slides  

---

## Slide 1: Title & Overview
* **Title:** Autonomous Search, Rescue & Evacuation in Hazardous Wildfire Terrain: From Classical Search (BFS, A*) to Modern Reinforcement Learning (Q-Learning, Actor-Critic)
* **Course:** Machine Learning (23MEE305)
* **Team:** 3 Students
* **Core Investigation:** The evolutionary trajectory from unguided and heuristic graph search to adaptive, closed-loop reinforcement learning under multi-objective hazard constraints.

---

## Slide 2: Two-Phase Mission Scenario & Disaster Environment
* **The Mission:** Complete sequential Search & Rescue operations in an intricate 15x15 canyon maze ($|S| = 450$ states):
  * **Phase 1 (Search & Ingress):** Launch from **Depot Base (0, 0)** $\to$ Penetrate rock barriers $\to$ Extract injured ranger trapped inside an enclosed interior canyon pocket at **(7, 2)**.
  * **Phase 2 (Evacuation & Egress):** Secure ranger onboard $\to$ Navigate escape corridors $\to$ Deliver patient to **Hospital Evacuation Base at (14, 14)**.
* **Environmental Hazards & Dilemmas:**
  * **Enclosed Canyon Pocket:** Ranger at (7, 2) is boxed on 3 sides by cliffs, with a single corridor opening at (7, 3).
  * **Fire Zone 1 (Entrance Gorge):** Rows 3–5, Col 5 (Core flame $T=1.0$, Smoke $T=0.5$).
  * **Fire Zone 2 (Evacuation Blaze):** Rows 9–12, Col 8 (Core flame $T=1.0$, Smoke $T=0.5$).
  * **The Core Tension:** Geometric straight-line paths cut straight into both fires. A blind agent incurs severe thermal burns, whereas risk-aware RL policies discover safe bypass routes.

---

## Slide 3: Algorithmic Spectrum & Implementation
* **Classical Graph Search:**
  * **BFS (Uninformed):** Expands 200 nodes; guarantees shortest hops (32 hops), but charges straight through both fires (Thermal Hazard = $4.50$).
  * **A* (Manhattan):** Prunes search to 73 nodes using distance heuristic $h(s)$, but shares BFS's fatal flaw (Hazard = $4.50$).
  * **A* (Risk-Aware):** Cost-augmented search $g(s) + \lambda \cdot T(s)$; discovers safe western bypass (34 hops, **Hazard = 0.00**).
* **Modern Reinforcement Learning (NumPy First Principles):**
  * **Q-Learning (Value-Based RL):** Off-policy Temporal Difference learning updating $Q(s, a)$ with $\epsilon$-greedy exploration.
  * **Actor-Critic (Policy-Based RL):** Advantage Actor-Critic (A2C) with Softmax Actor $\pi(a|s)$ and TD Critic $V(s)$.

---

## Slide 4: Empirical Benchmark Results

| Algorithm Paradigm | Nodes / Episodes | Runtime (s) | Path Hops | Thermal Hazard | Mission Success |
|---|---|---|---|---|---|
| **BFS (Uninformed Search)** | 200 nodes | 0.002 s | 32 hops | 4.50 (High) | 100% |
| **A* (Manhattan Heuristic)** | 73 nodes | 0.001 s | 32 hops | 4.50 (High) | 100% |
| **A* (Risk-Aware Heuristic)** | 86 nodes | 0.002 s | 34 hops | **0.00 (Zero)** | 100% |
| **Random Baseline Policy** | 50 episodes | 0.025 s | 220 steps | 0.00 (Lost) | 0.0% |
| **Greedy Distance Policy** | 1 step eval | 0.001 s | 220 steps | 0.00 (Trapped) | 0.0% |
| **Q-Learning (Value-based RL)**| 2000 episodes | 3.18 s | 38 hops | **0.00 (Zero)** | 100% |
| **Actor-Critic (A2C RL)** | 2000 episodes | 5.76 s | 38 hops | **0.00 (Zero)** | 100% |

---

## Slide 5: Potential-Based Reward Shaping & Ablation
* **Theoretical Grounding (Ng et al., 1999):**
  $$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$
* **Two-Phase Dynamic Potential:**
  $$\Phi(s) = \begin{cases} - [ \|s - \text{Ranger}\|_1 + \|\text{Ranger} - \text{Evac}\|_1 ] & \text{if } \neg\text{has\_ranger} \\ - \|s - \text{Evac}\|_1 & \text{if } \text{has\_ranger} \end{cases}$$
  * Guarantees strict potential continuity at pickup ($\Phi(\text{Ranger}, 0) = \Phi(\text{Ranger}, 1)$).
* **Ablation Findings:**
  * **Sparse Rewards:** Severe exploration bottleneck in 450-state MDP; random walks fail to solve sequential legs.
  * **Distance-Only Shaped:** Rapid convergence, but locks into the lethal 32-hop burn path.
  * **Full Risk-Aware Shaped:** Fast convergence AND zero thermal damage (Hazard = 0.00).

---

## Slide 6: Visual Analytics & Telemetry Animations
* **Publication Figures (`figures/`):**
  * `fig1_search_efficiency.png`: Nodes expanded vs computational latency vs hazard.
  * `fig2_rl_learning_curves.png`: Training episodic return and step convergence over 2000 episodes.
  * `fig3_reward_ablation.png`: Convergence speed vs thermal safety ablation curves.
  * `fig4_terrain_policy_value_heatmaps.png`: 4-panel terrain layout, value function $V(s)$, egress policy vector field, and trajectory overlay.
  * `fig5_bonus_robustness_noise.png`: Closed-loop RL vs open-loop search under atmospheric wind noise.
* **Telemetry Animated GIFs:**
  * `drone_comparison_search_vs_rl.gif`: Side-by-side comparison of A* Manhattan (burning) vs Q-Learning (safe detour).
  * `master_comparison_all_models.gif`: 4-quadrant live synchronized race of BFS, A* Manhattan, A* Risk-Aware, and Q-Learning.
  * Individual telemetry replays: `model_1_bfs.gif`, `model_2_astar_manhattan.gif`, `model_3_astar_risk_aware.gif`, `model_4_q_learning.gif`, `model_5_actor_critic.gif`.

---

## Slide 7: Bonus Feature (+5 Marks) — Robustness Under Wind Noise
* **The Experiment:** Benchmarking open-loop plan execution (A*) against closed-loop policy ($\pi^*(s)$) under wind slip $p_{\text{wind}} \in [0.0, 0.25]$.
* **The Verdict:**
  * **A* (Open-Loop Trajectory):** Fragile! A single wind gust drifts the UAV off the planned trajectory into cliffs or fire, dropping success steeply to $< 40\%$.
  * **Q-Learning (Closed-Loop Policy):** Resilient! The agent perceives its altered state at every step and executes real-time recovery maneuvers, sustaining $> 92\%$ mission success.

---

## Slide 8: Key Takeaways & Deliverables
* **Algorithmic Insight:** Classical search generates instant plans for static graphs; Reinforcement Learning is essential when environments involve multi-objective risk trade-offs and transition uncertainty.
* **Submission Package:**
  1. `wildfire_drone_env.py` (Gymnasium environment)
  2. `dynamic_wildfire_env.py` (Procedural & randomized dynamic hazard environment)
  3. `interactive_learning.py` (Human-in-the-Loop RL & Demonstration Learning)
  4. `interactive_game.py` (Playable game & interactive training web studio)
  5. `run_experiments.py` (Benchmarking suite & figure generator)
  6. `generate_demo_gif.py` (GIF telemetry generator)
  7. `Assignment_1_Wildfire_Drone.ipynb` (Executable Jupyter Notebook with Section 8)
  8. `Report_Assignment1_Wildfire_Drone_RL.docx` (Formal technical report)
  9. `Presentation_Slides.md` (Slide deck notes)
  10. `figures/` (5 publication PNGs + 8 animated GIFs)

---

## Slide 9: Interactive Game Studio & Human-in-the-Loop RL (Bonus Feature)
* **Blind AI Scout Benchmark & +2 Step Allowance:**
  * **RL Model Scouts in Secret:** On each procedural level, the RL agent secretly navigates and solves the maze first, establishing the optimal step count $S_{\text{RL}}$.
  * **Blind Human Challenge (+2 Allowance):** The path is NOT displayed and the step limit is hidden. The player must predict the minimum path independently, receiving a generous $+2$ step allowance ($S_{\text{max}} = S_{\text{RL}} + 2$).
  * **Zero Disclosure on Failure (Strict Retry):** If the player exceeds the step limit without rescuing the ranger, the level fails and restarts the **exact same layout**. Crucially, the AI's step count is **kept secret upon failure** so the player must figure out the optimal path on their own.
  * **Step Count Revealed Only on Victory:** The AI benchmark is revealed only when the player successfully beats/clears the level, advancing to a new procedural map.
* **Auto-Randomized Mazes on Successful Runs (Continuous Disaster Levels):**
  * When the human clears the level within the allowance, the game advances, procedurally generating new mountain ridges, canyon pockets, and wildfires with verified BFS reachability.
* **Cross-Maze Relational Q-Learning (Dual-System Generalization):**
  * Employs coordinate-invariant relational sensory states $(s_{\text{rel}})$ capturing relative goal bearing, local wall sensors, and adjacent wildfire detectors.
  * Knowledge acquired in one maze (dodging fire, navigating around obstacles) permanently persists into all future procedural mazes!
* **Learning from User Input (Interactive RL):**
  * **Real-time Interventions (DAgger):** User overrides autonomous steps using keyboard (WASD/Arrows); agent executes human actions and updates $Q(s, a)$ with demonstration bonuses.
  * **Human Preferences & Feedback (TAMER / RLHF):** Real-time scalar evaluations ($+1$ Praise / $-1$ Scold) directly shape action-values and modify TD updates.
* **Interactive Visual Studio & Game:**
  * Web-based game studio (`interactive_game.py`) featuring live flight HUD, streak badges, level clear banners, water cannon suppression, real-time policy arrow overlays (`↑`, `↓`, `←`, `→`), and multi-model arena comparisons.



