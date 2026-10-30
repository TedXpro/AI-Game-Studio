"""Data structures and kinematic update functions for the Raycast Survival Arena."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
import numpy as np
from numpy.typing import NDArray


class Action(IntEnum):
    """Discrete symmetric steering and thruster commands for the scout drone."""

    COAST = 0
    THRUST = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    THRUST_LEFT = 4
    THRUST_RIGHT = 5


class HitType(IntEnum):
    """Entity category detected by an individual LiDAR sensor ray."""

    NONE = 0
    WALL = 1
    HAZARD = 2
    CORE = 3


@dataclass(frozen=True)
class ArenaConfig:
    """Immutable physical, sensory, and reward parameters for the arena."""

    width: float = 800.0
    height: float = 600.0
    agent_radius: float = 14.0
    hazard_radius: float = 16.0
    core_radius: float = 12.0
    num_hazards: int = 5
    num_cores: int = 3
    hazard_speed: float = 1.75
    thrust_accel: float = 0.85
    linear_drag: float = 0.84
    turn_rate: float = 0.18
    max_speed: float = 5.0
    num_rays: int = 32
    max_ray_range: float = 350.0
    max_steps: int = 1000
    core_reward: float = 12.0
    crash_penalty: float = -35.0
    progress_scale: float = 0.05
    step_penalty: float = 0.0
    danger_margin: float = 52.0
    danger_penalty_scale: float = 0.30

    @property
    def observation_dim(self) -> int:
        """Returns the total length of the flat egocentric observation vector."""
        return self.num_rays * 4 + 5

    @property
    def action_dim(self) -> int:
        """Returns the total number of discrete control actions."""
        return len(Action)


@dataclass
class AgentState:
    """Mutable kinematic pose and velocity state of the autonomous scout drone."""

    position: NDArray[np.float32]
    velocity: NDArray[np.float32]
    heading: float

    def copy(self) -> AgentState:
        """Creates a deep copy of the agent state."""
        return AgentState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            heading=float(self.heading),
        )


@dataclass(frozen=True)
class RayReadout:
    """Geometric and categorical results of a 360-degree LiDAR sweep."""

    distances: NDArray[np.float32]
    hit_types: NDArray[np.int32]
    endpoints: NDArray[np.float32]


def wrap_angle(angle: float) -> float:
    """Wraps a radian angle into the canonical interval (-pi, pi].

    Args:
        angle: Input angle in radians.

    Returns:
        Wrapped angle in (-pi, pi].
    """
    wrapped = (angle + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if wrapped == -math.pi else wrapped


def decode_action_controls(action: int) -> tuple[float, float]:
    """Maps a discrete Action index into scalar thrust and turn inputs.

    Args:
        action: Integer action index in [0, 5].

    Returns:
        Tuple of (thrust_input, turn_direction) where thrust_input is in {0.0, 1.0}
        and turn_direction is in {-1.0, 0.0, 1.0}.
    """
    cmd = Action(action)
    thrust = 1.0 if cmd in (Action.THRUST, Action.THRUST_LEFT, Action.THRUST_RIGHT) else 0.0
    if cmd in (Action.TURN_LEFT, Action.THRUST_LEFT):
        turn = -1.0
    elif cmd in (Action.TURN_RIGHT, Action.THRUST_RIGHT):
        turn = 1.0
    else:
        turn = 0.0
    return thrust, turn


def advance_agent_kinematics(agent: AgentState, action: int, config: ArenaConfig) -> None:
    """Integrates one discrete tick of inertial rotation, thrust, and linear drag.

    Args:
        agent: Mutable drone state updated in place.
        action: Discrete control command selected for this tick.
        config: Physical constants governing acceleration, drag, and turn rate.
    """
    thrust, turn = decode_action_controls(action)
    agent.heading = wrap_angle(agent.heading + turn * config.turn_rate)
    forward = np.array(
        [math.cos(agent.heading), math.sin(agent.heading)],
        dtype=np.float32,
    )
    agent.velocity = (agent.velocity + forward * (thrust * config.thrust_accel)) * config.linear_drag
    speed = float(np.linalg.norm(agent.velocity))
    if speed > config.max_speed:
        agent.velocity *= config.max_speed / speed
    agent.position = agent.position + agent.velocity


def advance_hazards(
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    config: ArenaConfig,
) -> None:
    """Advances moving hazard mines and reflects them elastically off arena walls.

    Args:
        positions: Array of hazard centers of shape (num_hazards, 2), modified in place.
        velocities: Array of hazard velocities of shape (num_hazards, 2), modified in place.
        config: Arena boundary dimensions and hazard radius.
    """
    if positions.shape[0] == 0:
        return
    positions += velocities
    min_bound = config.hazard_radius
    max_x = config.width - config.hazard_radius
    max_y = config.height - config.hazard_radius

    hit_left = positions[:, 0] < min_bound
    hit_right = positions[:, 0] > max_x
    velocities[hit_left | hit_right, 0] *= -1.0
    positions[:, 0] = np.clip(positions[:, 0], min_bound, max_x)

    hit_top = positions[:, 1] < min_bound
    hit_bottom = positions[:, 1] > max_y
    velocities[hit_top | hit_bottom, 1] *= -1.0
    positions[:, 1] = np.clip(positions[:, 1], min_bound, max_y)


def is_wall_collision(agent: AgentState, config: ArenaConfig) -> bool:
    """Checks whether the drone hull intersects any of the four arena walls.

    Args:
        agent: Current drone state.
        config: Arena dimensions and agent radius.

    Returns:
        True if the drone touches or crosses an arena wall, else False.
    """
    r = config.agent_radius
    x, y = float(agent.position[0]), float(agent.position[1])
    return x <= r or x >= (config.width - r) or y <= r or y >= (config.height - r)


def compute_min_obstacle_clearance(
    agent: AgentState,
    hazards: NDArray[np.float32],
    config: ArenaConfig,
) -> float:
    """Computes minimum hull-to-surface clearance from walls and hazards.

    Args:
        agent: Current drone state.
        hazards: Active hazard positions of shape (num_hazards, 2).
        config: Arena dimensions and entity radii.

    Returns:
        Minimum clearance in pixels (negative or zero upon collision).
    """
    x, y = float(agent.position[0]), float(agent.position[1])
    wall_clear = min(x, config.width - x, y, config.height - y) - config.agent_radius
    if hazards.shape[0] == 0:
        return wall_clear
    dists = np.linalg.norm(hazards - agent.position[None, :], axis=1)
    haz_clear = float(np.min(dists)) - config.agent_radius - config.hazard_radius
    return min(wall_clear, haz_clear)


def find_colliding_circles(
    point: NDArray[np.float32],
    point_radius: float,
    centers: NDArray[np.float32],
    circle_radius: float,
) -> NDArray[np.bool_]:
    """Identifies which circles overlap with a query disc.

    Args:
        point: Query disc center of shape (2,).
        point_radius: Radius of the query disc.
        centers: Candidate circle centers of shape (N, 2).
        circle_radius: Radius shared by candidate circles.

    Returns:
        Boolean mask of shape (N,) indicating overlapping circles.
    """
    if centers.shape[0] == 0:
        return np.zeros(0, dtype=np.bool_)
    deltas = centers - point[None, :]
    dist_sq = np.sum(deltas * deltas, axis=1)
    threshold = point_radius + circle_radius
    return dist_sq <= (threshold * threshold)
