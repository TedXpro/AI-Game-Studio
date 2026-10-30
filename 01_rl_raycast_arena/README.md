# 01 — Raycast Survival Arena (Deep Reinforcement Learning)

An autonomous 2D survival and target-interception environment where a **Scout Drone** learns navigation, dynamic mine avoidance, and energy core harvesting **from scratch** using **Double Deep Q-Networks (Double DQN)** and **Egocentric 2D LiDAR Raycasting**.

---

## AI & Machine Learning Concepts

* **Egocentric Markov Decision Process (MDP)**:
  * **Observation Space ($\mathbb{R}^{53}$)**: 16 LiDAR rays spanning $360^\circ$ around the drone's nose $\times$ 3 occluded proximity channels (`Wall`, `Hazard`, `Core`) + 5 body-frame proprioception & target-compass features ($v_{\text{fwd}}, v_{\text{lat}}, \cos\Delta\phi, \sin\Delta\phi, d_{\text{core}}$).
  * **Symmetric Action Space ($|\mathcal{A}| = 6$)**: `COAST`, `THRUST`, `TURN_LEFT`, `TURN_RIGHT`, `THRUST_LEFT`, `THRUST_RIGHT`.
  * **Potential-Based Reward Shaping**: $+10.0$ per harvested core, $-10.0$ terminal crash penalty, $+0.06 \cdot (d_{t-1} - d_t)$ dense progress shaping, and $-0.015$ step urgency.
* **Double Deep Q-Network (`PyTorch`)**:
  * LayerNorm-stabilized MLP (`53 -> 128 -> 128 -> 6`) with orthogonal weight initialization.
  * **Pre-Allocated NumPy Replay Buffer** ($50,000$ transitions) breaking temporal correlation.
  * **Decoupled Action Selection & Evaluation**: Online network $Q_\theta$ selects $a^* = \arg\max_{a'} Q_\theta(s_{t+1}, a')$, while Polyak-averaged target network $Q_{\theta^-}$ evaluates $Q_{\theta^-}(s_{t+1}, a^*)$.

---

## Modular Architecture (All Files Strictly <= 250 LOC)

```text
01_rl_raycast_arena/
├── checkpoints/
│   └── best_dqn.pt                 # Trained champion Double-DQN weights
├── pyproject.toml                  # Pytest path configuration
├── requirements.txt                # Python dependencies
├── src/
│   ├── sim/
│   │   ├── entities.py             # Inertial kinematics, wall bounce & disc collisions
│   │   ├── raycast.py              # Vectorized 2D ray-wall & ray-circle LiDAR math
│   │   └── arena_env.py            # Headless reset() / step(action) MDP environment
│   ├── ai/
│   │   ├── q_network.py            # LayerNorm MLP Q-value network
│   │   ├── replay_buffer.py        # Contiguous NumPy circular experience memory
│   │   └── dqn_agent.py            # Double-DQN Bellman update, Polyak sync & checkpoints
│   ├── view/
│   │   └── visualizer.py           # Pygame-CE LiDAR geometry & live Q(s, a) advantage HUD
│   └── main.py                     # Headless turbo CLI & 60 FPS interactive telemetry lab
└── tests/
    └── test_arena_and_dqn.py       # Unit tests + automated <=250 LOC & zero-comment gate
```

---

## Quick Start

```powershell
cd 01_rl_raycast_arena

# 1. Watch the trained champion brain at 60 FPS with live Q(s, a) telemetry
python -m src.main --load checkpoints/best_dqn.pt --eval

# 2. Train a brand-new brain interactively (Press SPACE to toggle Turbo Mode)
python -m src.main

# 3. Train headlessly from the command line
python -m src.main --headless --episodes 350

# 4. Run unit & clean-code compliance tests
pytest -v
```

### Interactive Lab Hotkeys
* `SPACE` — Toggle **Turbo Training** vs. **60 FPS Live Watch**
* `E` — Toggle **Greedy Evaluation** ($\epsilon = 0$) vs. Exploration
* `R` — Toggle **LiDAR Ray Overlay**
* `S` / `L` — Save / Load `checkpoints/best_dqn.pt`
* `ESC` — Quit

---

## Training Benchmark (CPU — Intel Core i7-11800H)

| Stage | Epsilon ($\epsilon$) | Greedy Eval Avg Reward | Avg Cores Harvested / Episode |
| :--- | :-: | :-: | :-: |
| **Untrained (Episode 0)** | `1.000` | `-24.28` | `0.00` |
| **Episode 50** | `0.766` | `+1.73` | `0.67` |
| **Episode 100** | `0.378` | `+97.85` | `5.40` |
| **Champion (`best_dqn.pt`)** | `0.040` | **`+160.92`** | **`8.27`** |
