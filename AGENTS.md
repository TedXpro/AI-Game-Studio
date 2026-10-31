# AI Game Studio — Workspace Rules

Every project in **AI Game Studio** (`01_rl_raycast_arena`, `02_goap_tactical_squad`, `03_alphazero_quoridor`) must strictly obey all six engineering and educational standards below:

1. **Strict 250-Line File Cap**:
   - Every `.py`, `.hpp`, and `.cpp` source or test file must remain **strictly <= 250 lines of code** (target: 80–235 lines).
   - Handle growth via single-responsibility module decomposition, never by compressing formatting.

2. **Header Docstrings Only — Zero Inline Comments**:
   - Include clean Google-style docstrings (Python) or Doxygen `@brief`/`@param`/`@return` headers (C++20) on modules, classes/structs, and functions/methods.
   - **Never** write inline or end-of-line comments inside function bodies (`#` in Python source files or `//` inside C++ function bodies). Extract helper functions with descriptive names instead.

3. **Headless-First Layer Isolation (`sim/` <-> `ai/` <-> `view/`)**:
   - `sim/` implements pure deterministic physics, sensors, and rules with zero UI (`pygame`, `raylib`) or deep-learning (`torch`) imports.
   - `ai/` implements neural networks, RL trainers, GOAP planners, or MCTS trees with zero UI imports.
   - `view/` renders minimalist geometric state and live AI telemetry, supporting continuous/endless evaluation without artificial step resets.

4. **Progressive Milestone Checkpoints & Live Stage Hotkeys (`[1]`–`[4]`)**:
   - Every game must provide **4 evolutionary capability stages** selectable live via keyboard keys **`1`, `2`, `3`, `4`** in the visualizer (e.g., Stage 1 Untrained/Baseline -> Stage 2 Novice -> Stage 3 Skilled -> Stage 4 Champion/Full AI) plus structured progression logs (`checkpoints/training_history.json` or `.csv`) proving quantitative improvement.

5. **Interactive Learning & Telemetry Notebook (`notebooks/<project>_guide.ipynb`)**:
   - Every project must include a pre-executed Jupyter Notebook in `notebooks/` containing intuitive Markdown + LaTeX explanations, sensor/state visualizations, multi-panel progression/benchmark charts (`matplotlib`), and Stage 1–4 behavior comparisons importing from `src/` (or loading C++ telemetry CSVs) without duplicating engine code.

6. **Automated Compliance Gate & Atomic Git Commits**:
   - Every project's test suite must automatically verify the `<= 250` line limit and zero-inline-comment rule.
   - Commit changes in logical, separate conventional commits (`feat`, `test`, `docs`, `chore`) rather than monolithic dumps.
