# Engineering & Clean Code Standards

Every project in **AI Game Studio** adheres to four non-negotiable architectural and formatting rules designed for production readability, testability, and high-speed Machine Learning simulation.

---

## 1. Strict 250-Line File Cap

* Every source and header file must remain **strictly <= 250 lines of code** (target range: **80–220 lines**).
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
src/
├── sim/    # Pure environment rules, physics, sensors, and step(action) API. Zero UI imports.
├── ai/     # Neural networks, RL trainers, MCTS search trees, GOAP planners, and A*.
├── view/   # Lightweight geometric 2D renderer & live AI telemetry overlay.
└── main    # Entry point wiring headless turbo training and 60 FPS live evaluation.
```

1. `sim/` **never** imports `pygame`, `raylib`, or terminal rendering libraries.
2. Because `sim/` is pure logic, training loops can execute at **1,000x+ real-time speed** headlessly and seamlessly switch to **1x real-time visualization** to inspect agent behavior.
3. All randomness uses explicit, seeded random number generators (`np.random.Generator`, `torch.Generator`, `std::mt19937`) for reproducible benchmarks.
