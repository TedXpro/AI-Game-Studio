"""Headless Markov Decision Process environment for the Raycast Survival Arena."""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numpy.typing import NDArray

from src.sim.entities import (
    AgentState,
    ArenaConfig,
    RayReadout,
    advance_agent_kinematics,
    advance_hazards,
    compute_min_obstacle_clearance,
    find_colliding_circles,
    is_wall_collision,
)
from src.sim.raycast import build_observation, cast_sensor_rays, extract_danger_proximity


@dataclass(frozen=True)
class StepResult:
    """Immutable outcome of a single discrete MDP environment transition."""

    observation: NDArray[np.float32]
    reward: float
    terminated: bool
    truncated: bool
    cores_collected: int


@dataclass(frozen=True)
class ArenaSnapshot:
    """Read-only geometric state snapshot consumed by the optional visualizer."""

    agent: AgentState
    hazards: NDArray[np.float32]
    cores: NDArray[np.float32]
    rays: RayReadout
    step_count: int
    episode_cores: int
    episode_reward: float


class RaycastArenaEnv:
    """Headless 2D survival and target-harvesting environment with LiDAR perception."""

    def __init__(self, config: ArenaConfig | None = None, seed: int = 42) -> None:
        """Initializes the headless simulation arena with a seeded PRNG.

        Args:
            config: Physical and MDP settings; uses defaults when None.
            seed: Initial random seed for deterministic spawning.
        """
        self.config = config if config is not None else ArenaConfig()
        self._rng = np.random.default_rng(seed)
        self.agent = AgentState(
            position=np.array([self.config.width * 0.5, self.config.height * 0.5], dtype=np.float32),
            velocity=np.zeros(2, dtype=np.float32),
            heading=0.0,
        )
        self.hazards = np.zeros((self.config.num_hazards, 2), dtype=np.float32)
        self.hazard_velocities = np.zeros((self.config.num_hazards, 2), dtype=np.float32)
        self.cores = np.zeros((self.config.num_cores, 2), dtype=np.float32)
        self.last_rays = cast_sensor_rays(self.agent, self.hazards, self.cores, self.config)
        self._prev_danger = extract_danger_proximity(self.last_rays, self.config)
        self.step_count = 0
        self.episode_cores = 0
        self.episode_reward = 0.0

    def reset(self, seed: int | None = None) -> NDArray[np.float32]:
        """Resets the arena to a fresh episode and returns the initial observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        center = np.array([self.config.width * 0.5, self.config.height * 0.5], dtype=np.float32)
        heading = float(self._rng.uniform(-math.pi, math.pi))
        self.agent = AgentState(
            position=center,
            velocity=np.zeros(2, dtype=np.float32),
            heading=heading,
        )
        self._spawn_hazards()
        self._spawn_all_cores()
        self.step_count = 0
        self.episode_cores = 0
        self.episode_reward = 0.0
        self.last_rays = cast_sensor_rays(self.agent, self.hazards, self.cores, self.config)
        self._prev_danger = extract_danger_proximity(self.last_rays, self.config)
        return build_observation(
            self.agent, self.last_rays, self.cores, self.config, self._prev_danger
        )

    def step(self, action: int, endless: bool = False) -> StepResult:
        """Advances the simulation by one tick and computes reward and termination flags."""
        if action < 0 or action >= self.config.action_dim:
            raise ValueError(f"Action {action} outside valid range [0, {self.config.action_dim}).")

        prev_nearest = self._nearest_core_distance()
        advance_agent_kinematics(self.agent, action, self.config)
        advance_hazards(self.hazards, self.hazard_velocities, self.config)
        self.step_count += 1

        post_nearest = self._nearest_core_distance()
        reward = self.config.step_penalty + self.config.progress_scale * (
            prev_nearest - post_nearest
        )

        crashed = is_wall_collision(self.agent, self.config) or bool(
            np.any(
                find_colliding_circles(
                    self.agent.position,
                    self.config.agent_radius,
                    self.hazards,
                    self.config.hazard_radius,
                )
            )
        )

        collected_now = 0
        if crashed:
            reward += self.config.crash_penalty
        else:
            clearance = compute_min_obstacle_clearance(self.agent, self.hazards, self.config)
            if clearance < self.config.danger_margin:
                intrusion = 1.0 - max(0.0, clearance) / self.config.danger_margin
                reward -= self.config.danger_penalty_scale * (intrusion * intrusion)
            collected_now = self._collect_and_respawn_cores()
            reward += collected_now * self.config.core_reward

        self.episode_cores += collected_now
        self.episode_reward += reward
        terminated = crashed
        truncated = (not endless) and (not terminated) and (
            self.step_count >= self.config.max_steps
        )

        self.last_rays = cast_sensor_rays(self.agent, self.hazards, self.cores, self.config)
        obs = build_observation(
            self.agent, self.last_rays, self.cores, self.config, self._prev_danger
        )
        self._prev_danger = extract_danger_proximity(self.last_rays, self.config)
        return StepResult(
            observation=obs,
            reward=float(reward),
            terminated=terminated,
            truncated=truncated,
            cores_collected=collected_now,
        )

    def get_snapshot(self) -> ArenaSnapshot:
        """Exports an immutable snapshot of arena geometry for telemetry rendering.

        Returns:
            ArenaSnapshot reflecting the current tick state.
        """
        return ArenaSnapshot(
            agent=self.agent.copy(),
            hazards=self.hazards.copy(),
            cores=self.cores.copy(),
            rays=self.last_rays,
            step_count=self.step_count,
            episode_cores=self.episode_cores,
            episode_reward=self.episode_reward,
        )

    def _nearest_core_distance(self) -> float:
        """Computes Euclidean distance from the agent to the closest active energy core."""
        if self.cores.shape[0] == 0:
            return 0.0
        dists = np.linalg.norm(self.cores - self.agent.position[None, :], axis=1)
        return float(np.min(dists))

    def _sample_safe_point(
        self,
        margin: float,
        avoid_points: NDArray[np.float32],
        min_clearance: float,
    ) -> NDArray[np.float32]:
        """Samples a random arena coordinate maintaining clearance from existing entities."""
        for _ in range(100):
            candidate = np.array(
                [
                    self._rng.uniform(margin, self.config.width - margin),
                    self._rng.uniform(margin, self.config.height - margin),
                ],
                dtype=np.float32,
            )
            if avoid_points.shape[0] == 0:
                return candidate
            dists = np.linalg.norm(avoid_points - candidate[None, :], axis=1)
            if bool(np.all(dists >= min_clearance)):
                return candidate
        return candidate

    def _spawn_hazards(self) -> None:
        """Initializes moving hazards at safe starting distance from the center-spawned drone."""
        placed: list[NDArray[np.float32]] = [self.agent.position.copy()]
        margin = self.config.hazard_radius + 20.0
        for idx in range(self.config.num_hazards):
            avoid = np.stack(placed, axis=0)
            pos = self._sample_safe_point(margin, avoid, min_clearance=130.0)
            placed.append(pos)
            self.hazards[idx] = pos
            angle = float(self._rng.uniform(-math.pi, math.pi))
            self.hazard_velocities[idx] = np.array(
                [
                    math.cos(angle) * self.config.hazard_speed,
                    math.sin(angle) * self.config.hazard_speed,
                ],
                dtype=np.float32,
            )

    def _spawn_all_cores(self) -> None:
        """Spawns all energy cores away from walls, the agent, and active hazards."""
        placed: list[NDArray[np.float32]] = [self.agent.position.copy(), *list(self.hazards)]
        margin = self.config.core_radius + 64.0
        for idx in range(self.config.num_cores):
            avoid = np.stack(placed, axis=0)
            pos = self._sample_safe_point(margin, avoid, min_clearance=90.0)
            placed.append(pos)
            self.cores[idx] = pos

    def _collect_and_respawn_cores(self) -> int:
        """Detects collected cores, immediately respawns them safely, and returns the count."""
        hit_mask = find_colliding_circles(
            self.agent.position,
            self.config.agent_radius,
            self.cores,
            self.config.core_radius,
        )
        hit_indices = np.flatnonzero(hit_mask)
        margin = self.config.core_radius + 64.0
        for idx in hit_indices:
            other_cores = np.delete(self.cores, idx, axis=0)
            avoid = np.vstack((self.agent.position[None, :], self.hazards, other_cores))
            self.cores[idx] = self._sample_safe_point(margin, avoid, min_clearance=90.0)
        return int(hit_indices.shape[0])
