"""Unit tests for kinematics, 2D LiDAR raycasting, Double DQN, and code standards."""

from __future__ import annotations

import ast
import io
from pathlib import Path
import tokenize
import numpy as np
import pytest

from src.ai.dqn_agent import DQNAgent, DQNConfig
from src.sim.arena_env import RaycastArenaEnv
from src.sim.entities import Action, AgentState, ArenaConfig, HitType
from src.sim.raycast import build_observation, cast_sensor_rays
from src.view.visualizer import ArenaVisualizer, TelemetryFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_clean_code_and_layer_isolation_standards() -> None:
    """Verifies <=250 LOC, zero inline comments, and strict layer import isolation."""
    py_files = sorted(PROJECT_ROOT.rglob("*.py"))
    assert len(py_files) >= 8

    for file_path in py_files:
        source = file_path.read_text(encoding="utf-8")
        lines = source.splitlines()
        assert len(lines) <= 250, f"{file_path.name} has {len(lines)} lines (limit: 250)."

        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
        comments = [tok for tok in tokens if tok.type == tokenize.COMMENT]
        assert not comments, f"{file_path.name} contains prohibited inline comments."

        rel_parts = file_path.relative_to(PROJECT_ROOT).parts
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])

        if "sim" in rel_parts:
            assert "torch" not in imported
            assert "pygame" not in imported
        if "ai" in rel_parts:
            assert "pygame" not in imported


def test_raycast_analytical_distances_and_occlusion() -> None:
    """Checks exact wall distances, circle surface hits, and line-of-sight occlusion."""
    cfg = ArenaConfig(width=800.0, height=600.0, num_rays=16, max_ray_range=1000.0)
    agent = AgentState(
        position=np.array([100.0, 300.0], dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        heading=0.0,
    )
    empty = np.zeros((0, 2), dtype=np.float32)

    wall_readout = cast_sensor_rays(agent, empty, empty, cfg)
    assert wall_readout.distances[0] == pytest.approx(700.0, abs=1e-3)
    assert wall_readout.distances[8] == pytest.approx(100.0, abs=1e-3)
    assert int(wall_readout.hit_types[0]) == int(HitType.WALL)

    near_core = np.array([[200.0, 300.0]], dtype=np.float32)
    occluded_hazard = np.array([[350.0, 300.0]], dtype=np.float32)
    occ_readout = cast_sensor_rays(agent, occluded_hazard, near_core, cfg)

    expected_surface_dist = 200.0 - 100.0 - cfg.core_radius
    assert occ_readout.distances[0] == pytest.approx(expected_surface_dist, abs=1e-3)
    assert int(occ_readout.hit_types[0]) == int(HitType.CORE)

    obs = build_observation(agent, occ_readout, near_core, cfg)
    assert obs.shape == (cfg.observation_dim,)
    assert obs.dtype == np.float32


def test_environment_seed_determinism_and_core_collection() -> None:
    """Verifies identical seeded rollouts and positive reward upon core collection."""
    cfg = ArenaConfig()
    env_a = RaycastArenaEnv(cfg, seed=7)
    env_b = RaycastArenaEnv(cfg, seed=7)

    obs_a = env_a.reset(seed=77)
    obs_b = env_b.reset(seed=77)
    np.testing.assert_allclose(obs_a, obs_b)

    rng = np.random.default_rng(0)
    for _ in range(40):
        act = int(rng.integers(0, cfg.action_dim))
        res_a = env_a.step(act)
        res_b = env_b.step(act)
        np.testing.assert_allclose(res_a.observation, res_b.observation)
        assert res_a.reward == pytest.approx(res_b.reward)
        assert res_a.terminated == res_b.terminated
        if res_a.terminated:
            break

    env_collect = RaycastArenaEnv(cfg, seed=11)
    env_collect.reset(seed=11)
    env_collect.agent.position = np.array([400.0, 300.0], dtype=np.float32)
    env_collect.agent.heading = 0.0
    env_collect.hazards[:] = np.array([[80.0, 80.0]], dtype=np.float32)
    env_collect.cores[0] = np.array([405.0, 300.0], dtype=np.float32)

    step_res = env_collect.step(int(Action.THRUST))
    assert step_res.cores_collected >= 1
    assert step_res.reward > 5.0
    assert env_collect.cores.shape[0] == cfg.num_cores


def test_double_dqn_td_convergence_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    """Confirms Double-DQN converges on terminal target rewards and restores weights."""
    cfg = DQNConfig(
        obs_dim=ArenaConfig().observation_dim,
        action_dim=6,
        hidden_dim=64,
        learning_rate=5e-3,
        batch_size=16,
        warmup_steps=16,
        train_frequency=1,
        tau=0.2,
    )
    agent = DQNAgent(cfg, seed=123, device="cpu")
    fixed_state = np.ones(cfg.obs_dim, dtype=np.float32) * 0.25
    next_state = np.zeros(cfg.obs_dim, dtype=np.float32)

    losses: list[float] = []
    for _ in range(80):
        loss = agent.record_step(fixed_state, 2, 5.0, next_state, terminated=True)
        if loss is not None:
            losses.append(loss)

    assert losses[-1] < losses[0]
    _, q_vals = agent.select_action(fixed_state, greedy=True)
    assert float(q_vals[2]) == pytest.approx(5.0, abs=0.4)

    ckpt_path = tmp_path / "dqn_test.pt"
    agent.save_checkpoint(ckpt_path)
    restored = DQNAgent(cfg, seed=999, device="cpu")
    restored.load_checkpoint(ckpt_path)
    _, restored_q = restored.select_action(fixed_state, greedy=True)
    np.testing.assert_allclose(q_vals, restored_q, atol=1e-5)


def test_offscreen_visualizer_renders_telemetry_png(tmp_path: Path) -> None:
    """Verifies headless Pygame surface rendering and PNG export."""
    cfg = ArenaConfig()
    env = RaycastArenaEnv(cfg, seed=5)
    env.reset(seed=5)
    vis = ArenaVisualizer(cfg, offscreen=True)
    frame = TelemetryFrame(
        episode=12,
        epsilon=0.25,
        avg_reward=14.5,
        best_reward=32.0,
        steps_per_sec=2800.0,
        q_values=np.array([1.2, 4.8, 0.5, -0.3, 3.1, 2.0], dtype=np.float32),
        chosen_action=1,
        reward_history=[-8.0, -2.5, 6.0, 14.5, 22.0],
        mode_label="GREEDY EVAL",
    )
    out_file = tmp_path / "telemetry_test.png"
    vis.render(env.get_snapshot(), frame)
    vis.save_screenshot(out_file)
    vis.close()
    assert out_file.exists() and out_file.stat().st_size > 1000
