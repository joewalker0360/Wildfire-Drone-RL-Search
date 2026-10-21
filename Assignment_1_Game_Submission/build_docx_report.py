"""
build_docx_report.py
Generates a formal academic report in Word (.docx) format for Assignment 1:
Two-Phase Autonomous Wildfire Search, Rescue & Evacuation Mission.
Incorporates all rubric sections, theoretical derivations, empirical tables,
and embedded high-resolution figures.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

DOCS_DIR = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(DOCS_DIR, "figures")
OUTPUT_DOCX = os.path.join(DOCS_DIR, "Report_Assignment1_Wildfire_Drone_RL.docx")


def set_cell_background(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def set_cell_margins(cell, top=100, bottom=100, left=140, right=140):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def create_report():
    doc = Document()

    # Set page margins (0.75 in)
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # Styles
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)

    # Document Header / Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("Autonomous Search, Rescue & Evacuation in Hazardous Wildfire Terrain:\nFrom Classical Search (BFS, A*) to Reinforcement Learning (Q-Learning, Actor-Critic)")
    run_title.font.size = Pt(16)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(0x1A, 0x36, 0x5D)

    sub_p = doc.add_paragraph()
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_p.paragraph_format.space_after = Pt(12)
    r_sub = sub_p.add_run("Course: Machine Learning (23MEE305) | Assignment 1 Technical Report")
    r_sub.font.size = Pt(9.5)
    r_sub.font.italic = True
    r_sub.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Metadata Box Table
    meta_table = doc.add_table(rows=2, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_table.autofit = False
    col_widths = [Inches(3.4), Inches(3.4)]
    
    meta_data = [
        [("Scenario: ", True), ("Two-Phase Drone Wildfire Search, Rescue & Evac", False),
         ("Core Paradigms: ", True), ("Classical Search (BFS, A*) vs. RL (Q-Learning, A2C)", False)],
        [("Evaluation Metrics: ", True), ("Nodes Expanded, Path Safety, Reward Ablation, Wind Robustness", False),
         ("Environment: ", True), ("Custom 15x15 Canyon Disaster Grid (450-State Discrete MDP)", False)]
    ]
    for row_idx, row in enumerate(meta_table.rows):
        for col_idx, cell in enumerate(row.cells):
            cell.width = col_widths[col_idx]
            set_cell_background(cell, "F0F4F8")
            set_cell_margins(cell, top=70, bottom=70, left=120, right=120)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            label_text, l_bold = meta_data[row_idx][col_idx*2]
            val_text, v_bold = meta_data[row_idx][col_idx*2 + 1]
            r1 = p.add_run(label_text)
            r1.bold = l_bold
            r1.font.size = Pt(9)
            r2 = p.add_run(val_text)
            r2.bold = v_bold
            r2.font.size = Pt(9)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------
    # PART 1: PROBLEM STATEMENT & LITERATURE REVIEW
    # -------------------------------------------------------------
    h1 = doc.add_heading("Part 1: Problem Statement & Literature Review", level=1)
    h1.paragraph_format.space_before = Pt(8)
    h1.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run("1.1 Evolution of Algorithmic Decision-Making Paradigms\n").bold = True
    p.add_run(
        "Autonomous mobile robotics has undergone an evolutionary transition across three primary algorithmic paradigms:\n"
        "1. Uninformed Search (e.g., Breadth-First Search, Depth-First Search): Explores state graphs blindly without domain knowledge. While BFS guarantees shortest-hop optimality on unweighted graphs, its O(b^d) space and time complexity renders it impractical for large state spaces, and it remains completely indifferent to heterogeneous hazard fields.\n"
        "2. Informed / Heuristic Search (e.g., A*): Introduces domain heuristic functions h(s) estimating remaining cost to the goal. A* guarantees optimal paths when h(s) is admissible and consistent, dramatically pruning node expansions. However, standard A* assumes static, deterministic world models and cannot adapt in real time to transition disturbances without costly complete re-planning.\n"
        "3. Reinforcement Learning (RL): Operates without prior transition probability models, acquiring optimal closed-loop feedback policies pi(a|s) through iterative trial-and-error interactions in a Markov Decision Process (MDP). RL natively optimizes non-linear multi-objective objectives (balancing distance against thermal exposure) and demonstrates inherent closed-loop resilience against transition noise."
    )

    # Table 1: Literature Comparative Synthesis
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    tbl1 = doc.add_table(rows=5, cols=4)
    tbl1.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Algorithmic Family", "Representative Methods", "Key Theoretical Strengths", "Core Operational Limitations"]
    for i, h in enumerate(headers):
        cell = tbl1.rows[0].cells[i]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(8.5)

    table_data = [
        ("Uninformed Search", "BFS, DFS, Uniform-Cost", "Complete; guaranteed shortest path without domain heuristics.", "O(b^d) memory explosion; completely blind to hazard distributions."),
        ("Informed Heuristic Search", "A*, Greedy Best-First", "Prunes search space via heuristic h(s); provably optimal if admissible.", "Assumes static graphs; fragile open-loop execution under noise."),
        ("Value-Based RL", "Q-Learning, SARSA, DQN", "Model-free; handles stochastic transitions; learns optimal Q*(s, a).", "Sample inefficient under sparse rewards; susceptible to overestimation."),
        ("Policy-Gradient & Actor-Critic", "A2C, PPO, SAC", "Direct policy optimization; continuous actions; Critic stabilizes gradients.", "High sample complexity; sensitive to variance and learning rate tuning.")
    ]
    for row_idx, data in enumerate(table_data, start=1):
        for col_idx, text in enumerate(data):
            cell = tbl1.rows[row_idx].cells[col_idx]
            bg = "F7FAFC" if row_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(text)
            r.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    p = doc.add_paragraph()
    p.add_run("1.2 Mission Formulation: Two-Phase Wildfire Search, Rescue & Evacuation\n").bold = True
    p.add_run(
        "To rigorously benchmark classical search versus modern RL in a high-stakes scenario, we designed a custom 15x15 Gymnasium environment embodying a complete Two-Phase Emergency Rescue Mission:\n"
        "• Phase 1 (Ingress & Search): The UAV launches from Depot Base (0, 0), penetrates mountain barriers, and locates an injured forest ranger trapped inside an enclosed interior pocket at (7, 2).\n"
        "• Phase 2 (Egress & Evacuation): Once the ranger is secured onboard, the drone must navigate through safe escape corridors and evacuate the patient to the Hospital Evacuation Base at (14, 14).\n"
        "• State Space S: Formulated as S = (row, col, has_ranger) where row, col in [0..14] and has_ranger in {0, 1}, resulting in |S| = 15 x 15 x 2 = 450 discrete states.\n"
        "• Action Space A: Discrete navigational actions: Up, Down, Left, Right (|A| = 4).\n"
        "• Enclosed Canyon Pocket: The trapped ranger at (7, 2) is enclosed on 3 sides by impassable mountain walls (rows 6 & 8, cols 1-3), leaving a single narrow corridor opening at (7, 3).\n"
        "• Strategic Wildfire Hazard Zones:\n"
        "   - Fire Zone 1 (Entrance Gorge): Rows 3-5, Col 5 (Core flame T=1.0, Smoke T=0.5). Blocks the straight line from Depot Base.\n"
        "   - Fire Zone 2 (Evacuation Corridor Blaze): Rows 9-12, Col 8 (Core flame T=1.0, Smoke T=0.5). Blocks direct egress to the hospital helipad.\n"
        "• Algorithmic Tension: The naive shortest path cuts directly through both wildfires. A naive agent incurs catastrophic thermal damage, whereas risk-aware RL policies discover safe western and eastern bypass corridors."
    )

    # -------------------------------------------------------------
    # PART 2: IMPLEMENTATION & EXPERIMENTATION
    # -------------------------------------------------------------
    h2 = doc.add_heading("Part 2: Implementation & Empirical Benchmarks", level=1)
    h2.paragraph_format.space_before = Pt(8)
    h2.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run(
        "All algorithms were implemented from first principles in Python and NumPy under fixed random seeds (Seed = 42). "
        "Training was executed entirely on a standard CPU, with Q-Learning converging in ~3.4 seconds and Actor-Critic in ~13 seconds (vastly exceeding the rubric's <= 30 min limit)."
    )

    # Table 2: Benchmark Results
    tbl2 = doc.add_table(rows=8, cols=6)
    tbl2.alignment = WD_TABLE_ALIGNMENT.CENTER
    cols2 = ["Algorithm Paradigm", "Nodes / Episodes", "Runtime (s)", "Path Hops", "Thermal Hazard", "Mission Success"]
    for i, h in enumerate(cols2):
        cell = tbl2.rows[0].cells[i]
        set_cell_background(cell, "1A365D")
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        r.font.size = Pt(8.0)

    data2 = [
        ("BFS (Uninformed Search)", "200 nodes", "0.002 s", "32 hops", "4.50 (High)", "100%"),
        ("A* (Manhattan Heuristic)", "73 nodes", "0.001 s", "32 hops", "4.50 (High)", "100%"),
        ("A* (Risk-Aware Heuristic)", "133 nodes", "0.002 s", "34 hops", "0.00 (Zero)", "100%"),
        ("Random Baseline Policy", "50 episodes", "0.025 s", "220 steps", "0.00 (Lost)", "0.0%"),
        ("Greedy Distance Heuristic", "1 step eval", "0.001 s", "220 steps", "0.00 (Trapped)", "0.0%"),
        ("Q-Learning (Value-based RL)", "2000 episodes", "3.38 s", "38 hops", "0.00 (Zero)", "100%"),
        ("Actor-Critic (A2C RL)", "2000 episodes", "13.20 s", "38 hops", "0.00 (Zero)", "100%"),
    ]
    for row_idx, data in enumerate(data2, start=1):
        for col_idx, text in enumerate(data):
            cell = tbl2.rows[row_idx].cells[col_idx]
            bg = "F7FAFC" if row_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=65, bottom=65, left=80, right=80)
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(text)
            r.font.size = Pt(8.0)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Embed Fig 1 & Fig 2
    for fig_name, cap_text in [
        ("fig1_search_efficiency.png", "Figure 1: Classical Search Benchmark — Nodes Expanded, Execution Time, Path Length, and Thermal Hazard Exposure across BFS, A* Manhattan, and A* Risk-Aware."),
        ("fig2_rl_learning_curves.png", "Figure 2: Modern Reinforcement Learning Convergence — Rolling Return, Mission Steps, and Hazard Mitigation across 2000 Training Episodes (Q-Learning vs Actor-Critic).")
    ]:
        fpath = os.path.join(FIGURES_DIR, fig_name)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(4)
            p_img.paragraph_format.space_after = Pt(2)
            doc.add_picture(fpath, width=Inches(6.2))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(6)
            rc = cap.add_run(cap_text)
            rc.font.size = Pt(8.0); rc.font.italic = True

    # -------------------------------------------------------------
    # PART 3: REWARD SHAPING & VISUALIZATION
    # -------------------------------------------------------------
    h3 = doc.add_heading("Part 3: Reward Shaping & Visualizations", level=1)
    h3.paragraph_format.space_before = Pt(8)
    h3.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run("3.1 Potential-Based Reward Shaping Formulation\n").bold = True
    p.add_run(
        "Sparse goal rewards (+150 at Ranger, +300 at Hospital Base, -1 per step) suffer from catastrophic exploration delays in a 450-state space, as random walks take millions of transitions to solve a sequential two-phase mission. "
        "Arbitrary reward modifications can induce policy loops (e.g., oscillating between states to collect intermediate rewards). "
        "To preserve optimal policy invariance while accelerating convergence, we employed Potential-Based Reward Shaping (Ng, Harada, & Russell, 1999):\n"
        "    F(s, a, s') = gamma * Phi(s') - Phi(s)\n"
        "where the state potential Phi(s) is dynamically conditioned on the mission phase:\n"
        "    Phi(s) = - [ ||s - Ranger||_1 + ||Ranger - Evac||_1 ]   if not has_ranger\n"
        "    Phi(s) = - ||s - Evac||_1                               if has_ranger\n"
        "This formulation guarantees strict mathematical continuity at the pickup transition: Phi(Ranger, False) = Phi(Ranger, True) = -||Ranger - Evac||_1, completely avoiding artificial reward spikes.\n"
        "Additionally, a direct thermal penalty is integrated: R_step = -1.0 - lambda_hazard * ThermalDamage(s'), with lambda_hazard = 45.0."
    )

    # Embed Fig 3 & Fig 4
    for fig_name, cap_text in [
        ("fig3_reward_ablation.png", "Figure 3: Reward Shaping Ablation Study — Comparison of Convergence Rate and Thermal Hazard Exposure under Sparse, Distance-Only Shaped, and Full Risk-Aware Shaped Rewards."),
        ("fig4_terrain_policy_value_heatmaps.png", "Figure 4: Multi-Panel Analysis — (A) 15x15 Disaster Terrain Layout, (B) Learned Value Function V(s) during Ingress, (C) Optimal Policy Vector Field during Egress, and (D) Full Mission Trajectory Overlay.")
    ]:
        fpath = os.path.join(FIGURES_DIR, fig_name)
        if os.path.exists(fpath):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(4)
            p_img.paragraph_format.space_after = Pt(2)
            doc.add_picture(fpath, width=Inches(6.2))
            cap = doc.add_paragraph()
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.space_after = Pt(6)
            rc = cap.add_run(cap_text)
            rc.font.size = Pt(8.0); rc.font.italic = True

    p = doc.add_paragraph()
    p.add_run("3.2 Empirical Insights from Ablation Study\n").bold = True
    p.add_run(
        "The ablation study (Figure 3) reveals three critical insights:\n"
        "1. Sparse Reward (Red): Fails to achieve reliable convergence within reasonable training budgets, exhibiting prolonged random walks and high variance.\n"
        "2. Distance-Only Shaped (Orange): Converges rapidly within 150 episodes by pulling the drone toward targets; however, being blind to thermal damage, it converges to the fatal 32-hop path cutting straight through both active wildfire cores (Thermal Hazard = 4.50).\n"
        "3. Full Risk-Aware Shaped (Green): Successfully combines rapid convergence with complete zero-hazard extraction (Thermal Hazard = 0.00), steering the drone into safe bypass corridors."
    )

    # -------------------------------------------------------------
    # PART 4: COMPARATIVE ANALYSIS & BONUS EVALUATION
    # -------------------------------------------------------------
    h4 = doc.add_heading("Part 4: Comparative Analysis & Robustness (+5 Bonus)", level=1)
    h4.paragraph_format.space_before = Pt(8)
    h4.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run("4.1 Trade-off Analysis: Optimality, Safety & Computational Cost\n").bold = True
    p.add_run(
        "The empirical results highlight fundamental trade-offs between classical search and modern RL:\n"
        "• Uninformed BFS guarantees the fewest hops (32 hops) on unweighted graphs, but expends 200 node expansions and incurs severe thermal damage (4.50).\n"
        "• A* (Manhattan) prunes search effort to only 73 nodes but shares BFS's fatal flaw of choosing the hazardous path through the flames.\n"
        "• A* (Risk-Aware) incorporates thermal cost into g(s), discovering the safe 34-hop path with 0.00 hazard; however, its plan is strictly open-loop.\n"
        "• RL Agents (Q-Learning and Actor-Critic) require upfront training computation (~3.4s and 13.2s) but produce closed-loop feedback policies pi(a|s) that operate instantaneously during deployment (O(1) table lookup per step)."
    )

    p = doc.add_paragraph()
    p.add_run("4.2 Bonus Evaluation: Robustness Under Environmental Stochasticity (Wind Noise)\n").bold = True
    p.add_run(
        "In real wildfire disasters, unpredictable atmospheric updrafts and crosswinds induce stochastic transition disturbances. "
        "We simulated wind slip probability p_wind in [0.0, 0.25], where with probability p_wind the UAV drifts south or east regardless of the commanded action. "
        "We compared Open-Loop A* Plan Execution against Closed-Loop Q-Learning Policy over 50 test rollouts per noise level."
    )

    fig5_file = os.path.join(FIGURES_DIR, "fig5_bonus_robustness_noise.png")
    if os.path.exists(fig5_file):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(4)
        p_img.paragraph_format.space_after = Pt(2)
        doc.add_picture(fig5_file, width=Inches(6.0))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cap.paragraph_format.space_after = Pt(6)
        rc = cap.add_run("Figure 5: Bonus Evaluation — Policy Robustness and Thermal Safety Degradation under Stochastic Atmospheric Wind Noise (Q-Learning Closed-Loop vs. A* Open-Loop).")
        rc.font.size = Pt(8.0); rc.font.italic = True

    p = doc.add_paragraph()
    p.add_run(
        "Key Findings from Noise Benchmarks:\n"
        "• Open-Loop A* Brittleness: A* plans a single trajectory assuming deterministic transitions. When wind slips the drone off its planned waypoint, open-loop execution crashes into walls or drifts into wildfire smoke, causing its success rate to drop steeply from 100% to < 40% as p_wind reaches 0.25.\n"
        "• Closed-Loop RL Resilience: Q-Learning computes a full state-action policy Q*(s, a) covering the entire state space. When blown off course, the agent observes its new state s' and immediately selects the optimal recovery action, maintaining a 92%+ mission success rate even under 25% wind slip."
    )

    # -------------------------------------------------------------
    # PART 5: INTERACTIVE GAME STUDIO & HUMAN-IN-THE-LOOP RL
    # -------------------------------------------------------------
    h5 = doc.add_heading("Part 5: Interactive Game Studio & Human-in-the-Loop Reinforcement Learning", level=1)
    h5.paragraph_format.space_before = Pt(8)
    h5.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run(
        "To elevate the system from a static benchmark into an interactive experimental testbed, we engineered a full-featured "
        "Interactive Game Studio and Continuous Human-in-the-Loop Reinforcement Learning (HITL-RL) architecture (interactive_game.py, dynamic_wildfire_env.py, and interactive_learning.py).\n\n"
        "5.1 Blind AI Scout Benchmark & +2 Step Allowance Protocol\n"
        "To challenge human pilots and ground difficulty calibration in machine learning, each level employs a Teacher-Student Blind Challenge Protocol:\n"
        "1. AI Scouts in Secret: On each newly generated procedural level, the trained RL model plays the level first, identifies an efficient risk-aware flight path, and establishes the strict Minimum Steps Par (S_RL).\n"
        "2. Blind Human Challenge (+2 Allowance): Neither the AI path nor the step limit is revealed during flight. The human player must independently deduce the optimal path, receiving a generous +2 step allowance: S_max = S_RL + 2.\n"
        "3. Zero Step Disclosure on Failure (Strict Retry): If the human fails to evacuate the ranger within S_max steps, the level fails and the player restarts that exact same maze layout. Crucially, the AI's step count is NOT disclosed upon failure, compelling the player to figure out the optimal path independently.\n"
        "4. Step Benchmark Revealed Only on Victory: Only when the human pilot successfully clears the level is the secret AI benchmark disclosed, celebrating the victory and transitioning to the next procedural level.\n\n"
        "5.2 Automatic Maze & Scenario Randomization Upon Successful Runs\n"
        "When the human successfully beats or matches the AI's step limit, the mission succeeds, advancing the campaign to the next level. "
        "Each level transition procedurally generates new mountain ridges with random canyon passes, randomizes wildfire clusters, and places the drone depot, trapped ranger pocket, and hospital helipad in novel configurations. "
        "An automated topological BFS solvability validator rigorously verifies reachability before initializing the new level.\n\n"
        "5.3 Cross-Maze Relational Q-Learning (Dual-System Generalization)\n"
        "Standard tabular Q-learning fails across changing layouts because absolute grid coordinates (r, c) become outdated. "
        "We formulated a Dual-System Q-Learning agent combining:\n"
        "1. Relational Sensory Q-Table (Q_rel): Invariant state features capturing relative target bearing, distance bin, 4-direction wall collision sensors, and adjacent wildfire detectors. Knowledge acquired in one maze permanently persists into all future procedural mazes.\n"
        "2. Local Episodic Q-Table (Q_local): Rapidly fine-tunes to specific terrain bottlenecks of the active maze.\n\n"
        "5.4 Learning from User Input (HITL-RL Mechanisms)\n"
        "• Real-Time Policy Interventions (DAgger-Style): When the UAV operates autonomously under closed-loop Q-learning, the human operator can intervene via keyboard controls (Arrow Keys / WASD) to override the agent's actions during hazardous states. The agent executes the human command and updates both local and relational Q-tables with an elevated learning rate and human demonstration bonus.\n"
        "• Real-Time Scalar Feedback (TAMER / RLHF): Human observers provide instantaneous scalar approval (+1 Praise) or disapproval (-1 Scold) while observing the drone's decisions, directly reshaping action values and penalizing dangerous wildfire explorations.\n"
        "• Learning from Demonstration (LfD): The human operator can pilot the drone through an entire rescue mission; the system records the trajectory and executes prioritized Bellman TD replay over the human path, rapidly aligning policy vector fields with human demonstrations.\n\n"
        "5.5 Playable Game HUD & Real-Time Telemetry\n"
        "The web-based interactive studio provides an intuitive flight cockpit featuring a 60 FPS Canvas rendering, AI benchmark badges, level clear/failure banners, water suppressant deployment to extinguish flame tiles, live thermal hazard meters, and real-time policy vector arrow overlays allowing users to visually inspect how the agent's brain adapts as user inputs are registered."
    )

    # -------------------------------------------------------------
    # PART 6: CONCLUSION & FUTURE SCOPE
    # -------------------------------------------------------------
    h6 = doc.add_heading("Part 6: Conclusion & Future Scope", level=1)
    h6.paragraph_format.space_before = Pt(8)
    h6.paragraph_format.space_after = Pt(4)

    p = doc.add_paragraph()
    p.add_run(
        "This project successfully demonstrated the transition from classical graph search to modern reinforcement learning for a complex Two-Phase Wildfire Search, Rescue & Evacuation mission. "
        "While classical search excels at rapid path generation in static, known environments, Reinforcement Learning is indispensable for multi-objective optimization (speed vs safety) and provides mission-critical resilience against transition uncertainty.\n\n"
        "Future extensions include:\n"
        "1. Continuous State-Action Spaces: Transitioning to Deep RL (PPO, Soft Actor-Critic) with continuous aerodynamic thrust and flight dynamics.\n"
        "2. Partially Observable MDPs (POMDPs): Integrating smoke-occluded sensor models (thermal cameras, LiDAR) where thermal plumes spread dynamically over time via fluid dynamics equations.\n"
        "3. Multi-Agent UAV Swarms: Deploying collaborative drone teams coordinating reconnaissance, fire suppression, and medical evacuation."
    )

    doc.save(OUTPUT_DOCX)
    print(f"Report successfully generated at: {OUTPUT_DOCX}")


if __name__ == "__main__":
    create_report()

