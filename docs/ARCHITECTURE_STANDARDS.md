# Engineering & Clean Code Standards

Every project in **AI Game Studio** adheres to six non-negotiable architectural, pedagogical, and formatting rules designed for production readability, testability, and transparent AI/ML benchmarking.

---

## 1. Strict 250-Line File Cap

* Every `.py`, `.hpp`, and `.cpp` source and test file must remain **strictly <= 250 lines of code** (target range: **80–235 lines**).
* Growth is handled through **single-responsibility decomposition**, never by compressing formatting or sacrificing clarity.

---

## 2. Header Docstrings Only — Zero Inline Comments

* **Allowed**: File/module docstrings, class/struct headers, and function/method headers documenting purpose, inputs, and return values.
* **Prohibited**: Inline or end-of-line comments inside function bodies.
* **Why**: Anytime a block of logic feels complex enough to need an inline explanation, extract it into a small, descriptively named helper function with its own header comment.

### Python 3.12+ Function Standard

```python
def compute_td_target(
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Computes Bellman temporal-difference targets for a batch of transitions.

    Args:
        rewards: Immediate scalar rewards of shape (batch_size,).
        next_q_values: Maximal next-state Q-values from the target network.
        dones: Terminal flags (1.0 if episode ended, else 0.0).
        gamma: Discount factor in [0.0, 1.0].

    Returns:
        Target Q-value tensor of shape (batch_size,).
    """
    return rewards + gamma * next_q_values * (1.0 - dones)
```

### Modern C++20 Function Standard

```cpp
/**
 * @brief Computes the Manhattan heuristic distance between two grid coordinates.
 *
 * @param current Starting grid cell.
 * @param goal Target grid cell.
 * @return Heuristic movement cost estimate.
 */
[[nodiscard]] constexpr int EstimateHeuristic(GridPos current, GridPos goal) noexcept {
    return std::abs(current.x - goal.x) + std::abs(current.y - goal.y);
}
```

---

## 3. Headless-First Simulation Architecture

Every game separates simulation mathematics from visual presentation across three isolated layers:

```text
project_root/
├── checkpoints/  # Saved milestone weights (.pt) & structured telemetry (.json / .csv)
├── notebooks/    # Pre-executed Jupyter guide with math, sensor plots & learning curves
├── src/
│   ├── sim/      # Pure environment rules, physics, sensors, and step() API. Zero UI imports.
│   ├── ai/       # Neural networks, RL trainers, MCTS search trees, GOAP planners, and A*.
│   ├── view/     # Lightweight geometric 2D renderer, [1-4] stage hotkeys & AI telemetry HUD.
│   └── main      # Entry point wiring headless turbo runs and 60 FPS live evaluation.
└── tests/        # Unit tests + automated <=250 LOC & zero-inline-comment compliance gate.
```

1. `sim/` **never** imports `pygame`, `raylib`, `torch`, or terminal rendering libraries.
2. Evaluation mode runs **continuously without artificial step-limit resets** so long-horizon AI mastery can be observed directly.
3. All randomness uses explicit, seeded random number generators (`np.random.Generator`, `torch.Generator`, `std::mt19937`) for reproducible benchmarks.

---

## 4. Progressive Milestone Checkpoints & Live `[1]`–`[4]` Hotkeys

Every game persists **4 progressive AI capability stages** alongside structured training/benchmark logs (`checkpoints/training_history.json` or `.csv`):

* **`[1]` Stage 1 — Untrained / Naive Baseline**: Random weights or unguided patrol demonstrating raw problem difficulty.
* **`[2]` Stage 2 — Early Novice**: Initial survival or greedy single-step behavior.
* **`[3]` Stage 3 — Intermediate**: Competent navigation/planning before full convergence or team coordination.
* **`[4]` Stage 4 — Champion AI**: Fully converged neural checkpoint or full multi-agent planner.

Pressing keys **`1`, `2`, `3`, `4`** inside any project's live visualizer switches brains on the fly so progression is immediately visible.

---

## 5. Interactive Notebook Masterclass (`notebooks/`)

Every project ships a pre-executed Jupyter Notebook (`notebooks/<project>_guide.ipynb`) that imports directly from `src/` (or parses exported C++20 telemetry CSVs) to present:
1. Plain-English + mathematical derivations of the environment physics, sensors, and AI algorithms.
2. Visual inspections of what the AI perceives (LiDAR tensors, A\* threat fields, MCTS visit distributions).
3. Multi-panel `matplotlib` learning curves and Stage 1 $\rightarrow$ Stage 4 head-to-head benchmarks.

---

## 6. Automated Standards Gate & Atomic Commits

* Every project includes an automated test verifying `<= 250` lines per file, zero inline comments, and strict import isolation.
* All repository updates are recorded as atomic, layer-scoped conventional commits.
