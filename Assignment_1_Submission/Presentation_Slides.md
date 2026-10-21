# Presentation Slides: Evolution from Search to Reinforcement Learning
**Course:** Machine Learning (23MEE305) — Assignment 1  
**Project:** Two-Phase Autonomous Wildfire Search, Rescue & Evacuation Drone  
**Algorithms:** DFS, BFS, A*, Q-Learning, PPO, SAC  
**Deliverables:** Code, Visualizations, Telemetry Animated GIFs, 4-Page Report, Slides  

---

## Slide 1: Title & Overview
* **Title:** Autonomous Search, Rescue & Evacuation in Hazardous Wildfire Terrain: From Classical Search (DFS, BFS, A*) to Modern Reinforcement Learning (Q-Learning, PPO, SAC)
* **Course:** Machine Learning (23MEE305)
* **Team:** 3 Students
* **Core Investigation:** The evolutionary continuum from unguided and heuristic graph search to adaptive, closed-loop reinforcement learning under multi-objective hazard constraints.

---

## Slide 2: Two-Phase Mission Scenario & Disaster Environment
* **The Mission:** Sequential Search & Rescue operations in an intricate 15x15 canyon maze ($|S| = 450$ discrete states):
  * **Phase 1 (Search & Ingress):** Launch from **Depot Base (0, 0)** $\to$ Penetrate rock barriers $\to$ Extract injured ranger trapped inside an enclosed interior canyon pocket at **(7, 2)**.
  * **Phase 2 (Evacuation & Egress):** Secure ranger onboard $\to$ Navigate escape corridors $\to$ Deliver patient to **Hospital Evacuation Base at (14, 14)**.
* **Environmental Hazards & Dilemmas:**
  * **Enclosed Canyon Pocket:** Ranger at (7, 2) is walled in on 3 sides by cliffs, with a single corridor opening at (7, 3).
  * **Fire Zone 1 (Entrance Gorge):** Rows 3–5, Col 5 (Core flame $T=1.0$, Smoke $T=0.5$).
  * **Fire Zone 2 (Evacuation Blaze):** Rows 9–12, Col 8 (Core flame $T=1.0$, Smoke $T=0.5$).
  * **The Core Tension:** Geometric straight-line paths cut straight into both fires. A blind agent incurs severe thermal burns, whereas risk-aware RL policies discover safe bypass routes.

---

## Slide 3: Algorithmic Spectrum & Implementation
* **Classical Graph Search:**
  * **BFS (Uninformed):** Expands 200 nodes; guarantees shortest hops (32 hops), but charges straight through both fires (Thermal Hazard = $4.50$).
  * **A* (Manhattan):** Prunes search to 73 nodes using distance heuristic $h(s)$, but shares BFS's fatal flaw (Hazard = $4.50$).
  * **A* (Risk-Aware):** Cost-augmented search $g(s) + \lambda \cdot T(s)$; discovers safe western bypass (34 hops, **Hazard = 0.00**).
* **Modern Reinforcement Learning (Three Distinct Families in NumPy):**
  * **Q-Learning (Value-Based RL):** Off-policy Temporal Difference learning updating $Q(s, a)$ with $\epsilon$-greedy exploration.
  * **PPO (Policy-Gradient RL):** Clipped surrogate objective $L^{CLIP}(\theta)$ preventing destructive policy updates.
  * **SAC (Maximum-Entropy RL):** Soft Actor-Critic with entropy regularization $\alpha \mathcal{H}(\pi)$ promoting robust exploration.

---

## Slide 4: Empirical Benchmark Results

| Algorithm Paradigm | Nodes / Episodes | Runtime (s) | Path Hops | Thermal Hazard | Mission Success |
|---|---|---|---|---|---|
| **BFS (Uninformed Search)** | 200 nodes | 0.002 s | 32 hops | 4.50 (High) | 100% |
| **A* (Manhattan Heuristic)** | 73 nodes | 0.001 s | 32 hops | 4.50 (High) | 100% |
| **A* (Risk-Aware Heuristic)** | 86 nodes | 0.002 s | 34 hops | **0.00 (Zero)** | 100% |
| **Random Baseline Policy** | 50 episodes | 0.025 s | 220 steps | 0.00 (Lost) | 0.0% |
| **Greedy Distance Policy** | 1 step eval | 0.001 s | 220 steps | 0.00 (Trapped) | 0.0% |
| **Q-Learning (Value-based RL)**| 2000 episodes | 2.54 s | 38 hops | **0.00 (Zero)** | 100% |
| **PPO (Policy-Gradient RL)** | 2000 episodes | 5.80 s | 38 hops | **0.00 (Zero)** | 100% |
| **SAC (Soft Actor-Critic RL)**| 2000 episodes | 5.55 s | 40 hops | **0.00 (Zero)** | 100% |

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
  * **Full Risk-Aware Shaped:** Fast convergence AND zero thermal damage across all RL agents (Q-Learning, PPO, SAC).

---

## Slide 6: Visual Analytics & Telemetry Animations
* **Publication Figures (`figures/`):**
  * `fig1_search_efficiency.png`: Nodes expanded vs computational latency vs hazard.
  * `fig2_rl_learning_curves.png`: Training convergence over 2000 episodes comparing Q-Learning, PPO, and SAC.
  * `fig3_reward_ablation.png`: Convergence speed vs thermal safety ablation curves.
  * `fig4_terrain_policy_value_heatmaps.png`: 4-panel terrain layout, value function $V(s)$, egress policy vector field, and trajectory overlay across all models.
  * `fig5_bonus_robustness_noise.png`: Closed-loop RL vs open-loop search under atmospheric wind noise.
* **Telemetry Animated GIFs:**
  * `drone_comparison_search_vs_rl.gif`: Side-by-side comparison of A* Manhattan (burning) vs Q-Learning/PPO (safe detour).
  * `master_comparison_all_models.gif`: 4-quadrant live synchronized race of BFS, A* Manhattan, Q-Learning, and PPO.
  * Individual telemetry replays: `model_1_bfs.gif`, `model_2_astar_manhattan.gif`, `model_3_astar_risk_aware.gif`, `model_4_q_learning.gif`, `model_5_ppo.gif`, `model_6_sac.gif`.

---

## Slide 7: Bonus Feature (+5 Marks) — Robustness Under Wind Noise
* **The Experiment:** Benchmarking open-loop plan execution (A*) against closed-loop policy ($\pi^*(s)$) under wind slip $p_{\text{wind}} \in [0.0, 0.25]$.
* **The Verdict:**
  * **A* (Open-Loop Trajectory):** Fragile! A single wind gust drifts the UAV off the planned trajectory into cliffs or fire, dropping success steeply to $< 40\%$.
  * **Closed-Loop RL Policies:** Resilient! The agent perceives its altered state at every step and executes real-time recovery maneuvers, sustaining $> 92\%$ mission success.

---

## Slide 8: Key Takeaways & Deliverables
* **Algorithmic Insight:** Classical search generates instant plans for static graphs; Reinforcement Learning (Q-Learning, PPO, SAC) is essential when environments involve multi-objective risk trade-offs and transition uncertainty.
* **Submission Package:**
  1. `wildfire_drone_env.py` (Gymnasium environment)
  2. `run_experiments.py` (Benchmarking suite & figure generator)
  3. `generate_demo_gif.py` (GIF telemetry generator)
  4. `Assignment_1_Wildfire_Drone.ipynb` (Executable Jupyter Notebook)
  5. `Report_Assignment1_Wildfire_Drone_RL.docx` (Formal technical report)
  6. `Presentation_Slides.md` (Slide deck notes)
  7. `figures/` (5 publication PNGs + 8 animated GIFs)
