import numpy as np


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a unit vector; if zero-length, return original to avoid NaN."""
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def calculate_missile_position(
    initial_pos: np.ndarray,
    target_pos: np.ndarray,
    speed: float,
    t: float,
) -> np.ndarray:
    """Calculates missile position at time t, flying straight towards the target.

    The missile flies at constant speed along the straight line from initial_pos to target_pos.
    If the missile would pass the target within time t, clamp to target_pos.
    """
    direction = _unit_vector(target_pos - initial_pos)
    displacement = direction * speed * max(t, 0.0)
    to_target = target_pos - initial_pos
    if np.dot(displacement, direction) > np.linalg.norm(to_target):
        return target_pos.copy()
    return initial_pos + displacement


def calculate_uav_position(
    initial_pos: np.ndarray,
    velocity: float,
    direction: np.ndarray,
    t: float,
) -> np.ndarray:
    """Calculates UAV position at time t assuming constant velocity and direction."""
    dir_unit = _unit_vector(direction)
    return initial_pos + dir_unit * velocity * max(t, 0.0)


def calculate_decoy_trajectory(
    deploy_pos: np.ndarray,
    deploy_velocity: np.ndarray,
    g: float,
    t_after_deploy: float,
) -> np.ndarray:
    """Projectile motion for decoy after deployment.

    Simple ballistic trajectory with constant gravity acting in -Z.
    r(t) = r0 + v0*t + 0.5*a*t^2, where a = (0, 0, -g)
    """
    t_eff = max(t_after_deploy, 0.0)
    gravity = np.array([0.0, 0.0, -abs(g)], dtype=float)
    return deploy_pos + deploy_velocity * t_eff + 0.5 * gravity * (t_eff ** 2)


def calculate_smoke_cloud_position(
    detonation_pos: np.ndarray,
    sink_speed: float,
    t_after_detonation: float,
) -> np.ndarray:
    """Smoke cloud center sinks vertically with constant speed after detonation.

    The cloud center moves along -Z at sink_speed.
    """
    t_eff = max(t_after_detonation, 0.0)
    return detonation_pos + np.array([0.0, 0.0, -abs(sink_speed) * t_eff], dtype=float)
