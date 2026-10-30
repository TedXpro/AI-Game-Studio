# 01 — Raycast Survival Arena (Deep Reinforcement Learning)

An autonomous 2D survival and target-interception environment where agents learn navigation, hazard avoidance, and objective collection **from scratch** using **Deep Q-Networks (DQN)** and **2D Raycast Sensory Perception**.

---

## AI & Machine Learning Concepts

* **Markov Decision Process (MDP) Formulation**:
  * **Observation Space**: Normalized multi-ray distance & entity-type sensors (walls, moving hazards, energy cores) concatenated with agent velocity and heading.
  * **Action Space**: Discrete thruster and rotational steering commands.
  * **Reward Shaping**: Dense progress signals combined with sparse terminal penalties/bonuses to prevent reward hacking.
* **Deep Q-Learning (Double DQN)**:
  * Feedforward multilayer perceptron (`PyTorch`) estimating state-action values $Q(s, a)$.
  * **Experience Replay Buffer**: Decorrelates sequential transitions via uniform mini-batch sampling.
  * **Target Network Stabilization**: Periodic weight synchronization to prevent moving-target divergence.
  * **Epsilon-Greedy Exploration**: Smooth decay from exploratory behavior to greedy policy execution.

---

## Planned Modular Architecture (All Files <= 250 LOC)

```text
01_rl_raycast_arena/
├── checkpoints/                # Saved PyTorch model weights (.pt)
├── requirements.txt            # Python dependencies
└── src/
    ├── sim/
    │   ├── entities.py         # Agent, Hazard, and Target data structures & kinematics
    │   ├── raycast.py          # Vectorized 2D ray-segment & ray-circle intersection math
    │   └── arena_env.py        # Headless reset() / step(action) MDP environment
    ├── ai/
    │   ├── q_network.py        # PyTorch MLP architecture for Q-value estimation
    │   ├── replay_buffer.py    # Fixed-capacity circular experience replay memory
    │   └── dqn_agent.py        # Double-DQN training step, epsilon schedule & checkpointing
    ├── view/
    │   └── visualizer.py       # Minimal Pygame-CE geometric renderer + raycast & Q-value HUD
    └── main.py                 # CLI & interactive loop (Headless Turbo Train vs. 60 FPS Watch)
```

---

## Learning Roadmap

1. **Step 1 — Headless Physics & Kinematics (`sim/entities.py`, `sim/arena_env.py`)**: Build deterministic 2D movement, arena boundaries, moving hazards, and reward signals.
2. **Step 2 — Sensory Perception (`sim/raycast.py`)**: Implement 2D raycasting so the agent perceives only what its sensors detect, rather than omniscient coordinates.
3. **Step 3 — Neural Brain & Memory (`ai/q_network.py`, `ai/replay_buffer.py`, `ai/dqn_agent.py`)**: Implement Double DQN in PyTorch from scratch.
4. **Step 4 — Live Telemetry Visualizer (`view/visualizer.py`, `main.py`)**: Render vision rays, live Q-values, episode reward curves, and toggleable `1x` / `1000x` simulation speed.
