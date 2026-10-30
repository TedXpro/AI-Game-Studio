# 02 — Autonomous Tactical Squad (GOAP & Multi-Agent AI)

A real-time autonomous tactical simulation written in **Modern C++20** where specialized squad agents (Scout, Vanguard, Medic) coordinate without direct player movement control using **Goal-Oriented Action Planning (GOAP)**, a shared **Team Blackboard**, **Spatial A\* Pathfinding**, and **Steering Behaviors**.

---

## AI & Algorithmic Concepts

* **Goal-Oriented Action Planning (GOAP)**:
  * Inspired by *F.E.A.R.*, agents are not locked into brittle `if/else` trees.
  * Each action defines **symbolic preconditions**, **effects**, and a **procedural cost**.
  * Agents run backward **A\* search over the action space** in real time to synthesize multi-step plans (e.g., `MoveToAmmo -> Reload -> FlankTarget -> SuppressEnemy`).
* **Shared Squad Blackboard**:
  * Decentralized coordination layer publishing threat sightings, wounded ally alerts, and tactical cover reservations under fog-of-war.
* **Spatial A\* & Tactical Cost Fields**:
  * Grid/graph pathfinding augmented with danger weights so wounded or support agents naturally route through safe cover rather than exposed corridors.

---

## Planned Modular Architecture (All Files <= 250 LOC)

```text
02_goap_tactical_squad/
├── include/
│   ├── sim/
│   │   ├── world_types.hpp     # Coordinates, entities, agent roles, and tactical grid
│   │   └── simulation.hpp      # Deterministic tick loop, line-of-sight, and combat rules
│   ├── ai/
│   │   ├── world_state.hpp     # Bitmask symbolic state representation for fast planning
│   │   ├── goap_action.hpp     # Action preconditions, effects, and procedural cost interface
│   │   ├── goap_planner.hpp    # Regressive A* action-sequence synthesizer
│   │   ├── blackboard.hpp      # Shared squad memory & target/cover reservation table
│   │   └── pathfinder.hpp      # Tactical A* spatial navigation with threat-weighted costs
│   └── view/
│       └── renderer.hpp        # Minimalist visualizer & live agent plan/state inspector
└── src/
    ├── sim/                    # Simulation implementations (<= 250 LOC per file)
    ├── ai/                     # Planner, pathfinder, and role-goal implementations
    ├── view/                   # Telemetry & arena visualization
    └── main.cpp                # Simulation runner & sandbox scenario director
```

---

## Learning Roadmap

1. **Step 1 — Tactical Grid & Line-of-Sight (`sim/`)**: Build the obstacle map, agent stats, vision cones, and headless tick engine in C++20.
2. **Step 2 — Tactical A\* Navigation (`ai/pathfinder`)**: Implement threat-aware A\* pathfinding with custom priority queues.
3. **Step 3 — Symbolic GOAP Engine (`ai/world_state`, `ai/goap_planner`)**: Build bitmask world states and A\* planning over actions.
4. **Step 4 — Squad Coordination & Visualizer (`ai/blackboard`, `view/`)**: Connect Scouts, Medics, and Vanguards via the Blackboard and visualize active plans in real time.
