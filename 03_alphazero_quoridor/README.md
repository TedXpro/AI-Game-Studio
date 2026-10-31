# 03 — AlphaZero-Lite Quoridor (MCTS & Self-Play RL)

An adversarial strategy AI built in **Python & PyTorch** that masters **Quoridor**—a high-branching pawn-and-wall maze board game—using **Monte Carlo Tree Search (PUCT)** guided by a **Dual-Head Policy & Value Neural Network** trained exclusively via **AI-vs-AI Self-Play**.

---

## AI & Machine Learning Concepts

* **High-Branching Graph Game Mechanics**:
  * Players either advance their pawn or place horizontal/vertical walls to lengthen the opponent's shortest path—subject to a global **graph-reachability invariant** (walls may never completely seal off a player's goal row, verified via fast Breadth-First Search).
* **AlphaZero-Style PUCT Monte Carlo Tree Search**:
  * Balances exploitation $Q(s, a)$ with neural-prior-guided exploration $U(s, a) = c_{\text{puct}} \cdot P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$.
  * Eliminates random rollouts by evaluating leaf states directly with the Value head $v_\theta(s) \in [-1, 1]$ and masking illegal moves before normalizing Policy priors $p_\theta(s)$.
* **Automated Self-Play & Arena Gating**:
  * Generates training trajectories via temperature-controlled self-play, trains $(\mathbf{p}_\theta, v_\theta)$ on `(state, mcts_visit_distribution, match_outcome)` tuples, and promotes candidate checkpoints only when they defeat the reigning champion above a win-rate threshold.

---

## Planned Modular Architecture (All Files <= 250 LOC)

```text
03_alphazero_quoridor/
├── checkpoints/                # Milestone & Champion self-play weights (.pt) + history (.json)
├── notebooks/
│   └── 03_alphazero_guide.ipynb # Interactive PUCT math, policy heatmaps & Elo progression
├── requirements.txt            # Python dependencies
└── src/
    ├── sim/
    │   ├── graph_path.py       # Fast BFS shortest-path & legal wall-placement reachability check
    │   └── quoridor_env.py     # Canonical board state, legal action masks, and tensor encoding
    ├── ai/
    │   ├── policy_value_net.py # Dual-head PyTorch CNN (Policy logits + scalar Value head)
    │   ├── mcts.py             # PUCT search tree with legal-action masking & Dirichlet noise
    │   └── self_play.py        # Self-play trajectory generator, training loop & arena gating
    ├── view/
    │   └── board_view.py       # Minimal Pygame-CE board, wall preview & live MCTS win-rate bar
    └── main.py                 # Train via self-play, watch AI-vs-AI, or challenge the Champion AI
```

---

## Learning Roadmap

1. **Step 1 — Quoridor Rules & Graph Reachability (`sim/`)**: Encode board state, jump mechanics, legal action bitmasks, and BFS wall legality checks (starting on a fast 5x5 / 7x7 board for rapid laptop training, scalable to 9x9).
2. **Step 2 — Pure MCTS Baseline (`ai/mcts.py`)**: Build and verify the PUCT search tree against shortest-path heuristic bots.
3. **Step 3 — Dual-Head Neural Network (`ai/policy_value_net.py`)**: Construct the multi-channel spatial tensor encoder and Policy-Value CNN in PyTorch.
4. **Step 4 — Self-Play Loop, `[1]`–`[4]` Generation Hotkeys & Notebook (`ai/self_play.py`, `view/board_view.py`, `notebooks/`)**: Train generations via self-play, toggle `[1]`–`[4]` self-play milestones live in the board viewer, and plot win-rate/Elo curves in the companion notebook.
