# 01 — Raycast Survival Arena (Deep Reinforcement Learning)

An autonomous 2D survival and target-interception environment where a **Scout Drone** learns navigation, dynamic mine avoidance, and energy core harvesting **from scratch** using **Double Deep Q-Networks (Double DQN)** and **Egocentric 2D LiDAR Raycasting**.

---

## AI & Machine Learning Concepts

* **Egocentric Markov Decision Process (MDP)**:
  * **Observation Space ($\mathbb{R}^{133}$)**: 32 LiDAR rays spanning $360^\circ$ around the drone's nose $\times$ 3 occluded proximity channels (`Wall`, `Hazard`, `Core`) + 32 Doppler obstacle closing-velocity rates ($\Delta\text{proximity}_t$) + 5 body-frame proprioception & target-compass features ($v_{\text{fwd}}, v_{\text{lat}}, \cos\Delta\phi, \sin\Delta\phi, d_{\text{core}}$).
  * **Symmetric Action Space ($|\mathcal{A}| = 6$)**: `COAST`, `THRUST`, `TURN_LEFT`, `TURN_RIGHT`, `THRUST_LEFT`, `THRUST_RIGHT`.
  * **Safety-Aware Reward Shaping**: $+12.0$ per harvested core, $-35.0$ terminal crash penalty, smooth close-clearance repulsion buffer, and $+0.05 \cdot (d_{t-1} - d_t)$ potential-based progress shaping.
* **Dueling Double Deep Q-Network (`PyTorch`)**:
  * LayerNorm-stabilized Dueling MLP (`133 -> 160 -> 160 -> V(s) + A(s, a)`) decomposing state danger from steering advantage.
  * **Pre-Allocated NumPy Replay Buffer** ($60,000$ transitions) breaking temporal correlation.
  * **Decoupled Action Selection & Evaluation**: Online network $Q_\theta$ selects $a^* = \arg\max_{a'} Q_\theta(s_{t+1}, a')$, while Polyak-averaged target network $Q_{\theta^-}$ evaluates $Q_{\theta^-}(s_{t+1}, a^*)$.

---

## Modular Architecture (All Files Strictly <= 250 LOC)

```text
01_rl_raycast_arena/
├── checkpoints/
│   ├── milestone_1_ep000.pt        # [1] Stage 1: Untrained baseline
│   ├── milestone_2_ep035.pt        # [2] Stage 2: Early novice (wall avoidance)
│   ├── milestone_3_ep090.pt        # [3] Stage 3: Skilled core harvester
│   ├── best_dqn.pt                 # [4] Stage 4: Champion Dueling-Doppler brain
│   └── training_history.json       # Full 450-episode progression telemetry
├── notebooks/
│   └── 01_rl_arena_guide.ipynb     # Pre-executed interactive math & plotting guide
├── pyproject.toml                  # Pytest path configuration
├── requirements.txt                # Python dependencies
├── src/
│   ├── sim/
│   │   ├── entities.py             # Inertial kinematics, wall bounce & disc collisions
│   │   ├── raycast.py              # Vectorized 2D ray-wall, ray-circle & Doppler math
│   │   └── arena_env.py            # Headless reset() / step(action) MDP environment
│   ├── ai/
│   │   ├── q_network.py            # LayerNorm Dueling MLP Q-value network
│   │   ├── replay_buffer.py        # Contiguous NumPy circular experience memory
│   │   ├── dqn_agent.py            # Double-DQN Bellman update, Polyak sync & checkpoints
│   │   └── trainer.py              # Headless trainer, milestone saver & JSON logger
│   ├── view/
│   │   └── visualizer.py           # Pygame-CE LiDAR geometry & live Q(s, a) advantage HUD
│   └── main.py                     # Headless turbo CLI & [1-4] stage interactive lab
└── tests/
    └── test_arena_and_dqn.py       # Unit tests + automated <=250 LOC & zero-comment gate
```

---

## Quick Start

```powershell
cd 01_rl_raycast_arena

# 1. Watch the Stage 4 Champion brain in endless 60 FPS mode (press 1-4 to compare stages!)
python -m src.main --eval

# 2. Launch directly at a specific evolutionary stage (1 = Ep 0, 2 = Ep 35, 3 = Ep 90, 4 = Champion)
python -m src.main --stage 1

# 3. Train a brand-new brain interactively (Press SPACE to toggle Turbo Mode)
python -m src.main

# 4. Train headlessly and regenerate all 4 milestones + training_history.json
python -m src.main --headless --episodes 450

# 5. Open the interactive notebook (VS Code / Jupyter) or run unit tests
pytest -v
```

### Interactive Lab Hotkeys
* `1` / `2` / `3` / `4` — Hot-swap **Stage 1 (Ep 0)** $\rightarrow$ **Stage 2 (Ep 35)** $\rightarrow$ **Stage 3 (Ep 90)** $\rightarrow$ **Stage 4 (Champion)** live on screen!
* `SPACE` — Toggle **Turbo Training** vs. **60 FPS Live Watch**
* `E` — Toggle **Greedy Evaluation** ($\epsilon = 0$) vs. Exploration
* `R` — Toggle **LiDAR Ray Overlay**
* `S` / `L` — Save / Load `checkpoints/best_dqn.pt`
* `ESC` — Quit

---

## 100-Episode Unseen Benchmark (CPU — Intel Core i7-11800H)

| Pilot (100 Unseen Seeds) | Avg Score | Avg Cores / Ep | Wall Crash Rate | Full 1,000-Step Survival | Best Endless Run |
| :--- | :-: | :-: | :-: | :-: | :-: |
| **Random Baseline** | `-40.96` | `0.20` | `63%` | `0%` | `1` core (`2.1s`) |
| **Rule-Based `if/else` Bot** | `+135.00` | `7.88` | `0%` | `27%` | — |
| **v1 Single-Frame DQN** | `+112.21` | `5.90` | `8%` | `2%` | `14` cores (`13.2s`) |
| **v2 Dueling-Doppler (`best_dqn.pt`)** | **`+193.18`** | **`10.64`** | **`0%`** | **`25%`** | **`52` cores (`48.4s`)** |
