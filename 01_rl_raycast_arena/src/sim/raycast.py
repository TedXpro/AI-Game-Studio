"""Vectorized 2D LiDAR raycasting and egocentric observation encoding."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

from src.sim.entities import AgentState, ArenaConfig, HitType, RayReadout


def compute_ray_directions(heading: float, num_rays: int) -> NDArray[np.float32]:
    """Generates evenly spaced 2D unit direction vectors anchored to agent heading.

    Args:
        heading: Agent orientation angle in radians.
        num_rays: Number of rays spanning 2*pi radians.

    Returns:
        Unit direction vectors of shape (num_rays, 2).
    """
    offsets = np.linspace(0.0, 2.0 * math.pi, num=num_rays, endpoint=False, dtype=np.float32)
    angles = heading + offsets
    return np.stack((np.cos(angles), np.sin(angles)), axis=1).astype(np.float32)


def intersect_rays_with_walls(
    origin: NDArray[np.float32],
    directions: NDArray[np.float32],
    width: float,
    height: float,
) -> NDArray[np.float32]:
    """Computes ray distances to the four axis-aligned arena boundary walls.

    Args:
        origin: Ray origin coordinates of shape (2,).
        directions: Unit ray direction vectors of shape (num_rays, 2).
        width: Arena width along the x-axis.
        height: Arena height along the y-axis.

    Returns:
        Non-negative wall hit distances of shape (num_rays,).
    """
    dx = directions[:, 0]
    dy = directions[:, 1]

    tx = np.full(directions.shape[0], np.inf, dtype=np.float32)
    pos_x = dx > 1e-8
    neg_x = dx < -1e-8
    tx[pos_x] = (width - origin[0]) / dx[pos_x]
    tx[neg_x] = (0.0 - origin[0]) / dx[neg_x]

    ty = np.full(directions.shape[0], np.inf, dtype=np.float32)
    pos_y = dy > 1e-8
    neg_y = dy < -1e-8
    ty[pos_y] = (height - origin[1]) / dy[pos_y]
    ty[neg_y] = (0.0 - origin[1]) / dy[neg_y]

    return np.maximum(0.0, np.minimum(tx, ty)).astype(np.float32)


def intersect_rays_with_circles(
    origin: NDArray[np.float32],
    directions: NDArray[np.float32],
    centers: NDArray[np.float32],
    radius: float,
) -> NDArray[np.float32]:
    """Computes nearest analytical ray-circle intersection distances across circles.

    Args:
        origin: Ray origin coordinates of shape (2,).
        directions: Unit ray direction vectors of shape (num_rays, 2).
        centers: Circle center positions of shape (num_circles, 2).
        radius: Shared radius of the target circles.

    Returns:
        Nearest hit distance per ray of shape (num_rays,), or inf on a miss.
    """
    num_rays = directions.shape[0]
    if centers.shape[0] == 0:
        return np.full(num_rays, np.inf, dtype=np.float32)

    rel = centers - origin[None, :]
    proj = directions @ rel.T
    dist_sq = np.sum(rel * rel, axis=1)[None, :]
    radius_sq = radius * radius
    perp_sq = dist_sq - proj * proj

    half_chord = np.sqrt(np.maximum(0.0, radius_sq - perp_sq))
    inside = dist_sq <= radius_sq
    t_entry = np.where(inside, 0.0, proj - half_chord)
    valid = (perp_sq <= radius_sq) & ((proj + half_chord) >= 0.0) & (t_entry >= 0.0)
    distances = np.where(valid, t_entry, np.inf)
    return np.min(distances, axis=1).astype(np.float32)


def cast_sensor_rays(
    agent: AgentState,
    hazards: NDArray[np.float32],
    cores: NDArray[np.float32],
    config: ArenaConfig,
) -> RayReadout:
    """Computes closest occluded intersections for all egocentric LiDAR rays.

    Args:
        agent: Current pose and velocity of the scout drone.
        hazards: Hazard center coordinates of shape (num_hazards, 2).
        cores: Energy core center coordinates of shape (num_cores, 2).
        config: Arena dimensions, radii, and raycast range settings.

    Returns:
        RayReadout containing clipped distances, hit categories, and ray endpoints.
    """
    directions = compute_ray_directions(agent.heading, config.num_rays)
    best_dist = np.full(config.num_rays, config.max_ray_range, dtype=np.float32)
    hit_types = np.full(config.num_rays, int(HitType.NONE), dtype=np.int32)

    wall_dist = intersect_rays_with_walls(agent.position, directions, config.width, config.height)
    wall_mask = wall_dist < best_dist
    best_dist = np.where(wall_mask, wall_dist, best_dist)
    hit_types = np.where(wall_mask, int(HitType.WALL), hit_types)

    hazard_dist = intersect_rays_with_circles(
        agent.position, directions, hazards, config.hazard_radius
    )
    hazard_mask = hazard_dist < best_dist
    best_dist = np.where(hazard_mask, hazard_dist, best_dist)
    hit_types = np.where(hazard_mask, int(HitType.HAZARD), hit_types)

    core_dist = intersect_rays_with_circles(agent.position, directions, cores, config.core_radius)
    core_mask = core_dist < best_dist
    best_dist = np.where(core_mask, core_dist, best_dist)
    hit_types = np.where(core_mask, int(HitType.CORE), hit_types)

    endpoints = agent.position[None, :] + directions * best_dist[:, None]
    return RayReadout(
        distances=best_dist.astype(np.float32),
        hit_types=hit_types,
        endpoints=endpoints.astype(np.float32),
    )


def compute_nearest_core_features(
    agent: AgentState,
    cores: NDArray[np.float32],
    config: ArenaConfig,
) -> tuple[float, float, float]:
    """Computes body-frame bearing and normalized distance to the closest energy core.

    Args:
        agent: Current drone state.
        cores: Energy core coordinates of shape (num_cores, 2).
        config: Arena dimensions for diagonal distance normalization.

    Returns:
        Tuple of (cos_bearing, sin_bearing, normalized_distance).
    """
    if cores.shape[0] == 0:
        return 1.0, 0.0, 0.0
    deltas = cores - agent.position[None, :]
    dists = np.linalg.norm(deltas, axis=1)
    nearest_idx = int(np.argmin(dists))
    dist = float(dists[nearest_idx])
    if dist < 1e-6:
        return 1.0, 0.0, 0.0

    ux, uy = float(deltas[nearest_idx, 0]) / dist, float(deltas[nearest_idx, 1]) / dist
    cos_h, sin_h = math.cos(agent.heading), math.sin(agent.heading)
    cos_bearing = ux * cos_h + uy * sin_h
    sin_bearing = -ux * sin_h + uy * cos_h
    diag = math.hypot(config.width, config.height)
    return cos_bearing, sin_bearing, min(1.0, dist / diag)


def build_observation(
    agent: AgentState,
    readout: RayReadout,
    cores: NDArray[np.float32],
    config: ArenaConfig,
) -> NDArray[np.float32]:
    """Encodes LiDAR proximity channels and egocentric proprioception into a flat vector.

    Args:
        agent: Current pose and velocity of the drone.
        readout: Current LiDAR sweep results.
        cores: Active energy core positions.
        config: Arena configuration parameters.

    Returns:
        Normalized float32 observation vector of shape (observation_dim,).
    """
    proximity = np.clip(1.0 - (readout.distances / config.max_ray_range), 0.0, 1.0)
    ray_channels = np.zeros((config.num_rays, 3), dtype=np.float32)
    ray_channels[readout.hit_types == int(HitType.WALL), 0] = proximity[
        readout.hit_types == int(HitType.WALL)
    ]
    ray_channels[readout.hit_types == int(HitType.HAZARD), 1] = proximity[
        readout.hit_types == int(HitType.HAZARD)
    ]
    ray_channels[readout.hit_types == int(HitType.CORE), 2] = proximity[
        readout.hit_types == int(HitType.CORE)
    ]

    cos_h, sin_h = math.cos(agent.heading), math.sin(agent.heading)
    vx, vy = float(agent.velocity[0]), float(agent.velocity[1])
    v_forward = np.clip((vx * cos_h + vy * sin_h) / config.max_speed, -1.0, 1.0)
    v_lateral = np.clip((-vx * sin_h + vy * cos_h) / config.max_speed, -1.0, 1.0)
    cos_b, sin_b, norm_dist = compute_nearest_core_features(agent, cores, config)

    self_features = np.array([v_forward, v_lateral, cos_b, sin_b, norm_dist], dtype=np.float32)
    return np.concatenate((ray_channels.ravel(), self_features)).astype(np.float32)
