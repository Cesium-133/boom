"""
几何计算工具模块

提供导弹拦截烟幕遮蔽分析所需的几何计算功能，包括：
- 点到直线距离计算
- 轨迹点生成
- 垂足计算
- 目标点生成等

"""

from __future__ import annotations
from typing import List, Tuple, Callable
from math import sqrt, cos, sin, radians
from functools import lru_cache
import numpy as np

# 尝试导入Numba，如果不可用则使用标准实现
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # 定义空的装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

Vector3 = Tuple[float, float, float]

# =============================================================================
# Numba优化的核心计算函数
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _distance_between_numba(x0: float, y0: float, z0: float, 
                               x1: float, y1: float, z1: float) -> float:
        """计算两点间的欧氏距离 - Numba优化版本"""
        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        return sqrt(dx * dx + dy * dy + dz * dz)

    @jit(nopython=True, cache=True, fastmath=True)
    def _foot_of_perpendicular_numba(px: float, py: float, pz: float,
                                    sx: float, sy: float, sz: float,
                                    ex: float, ey: float, ez: float):
        """计算点到线段的垂足坐标 - Numba优化版本"""
        seg_x = ex - sx
        seg_y = ey - sy
        seg_z = ez - sz
        
        segment_length_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
        
        if segment_length_sq == 0.0:
            return (sx, sy, sz)
        
        to_x = px - sx
        to_y = py - sy
        to_z = pz - sz
        
        t = (to_x * seg_x + to_y * seg_y + to_z * seg_z) / segment_length_sq
        t = max(0.0, min(1.0, t))
        
        foot_x = sx + t * seg_x
        foot_y = sy + t * seg_y
        foot_z = sz + t * seg_z
        
        return (foot_x, foot_y, foot_z)

    @jit(nopython=True, cache=True, fastmath=True)
    def _get_circle_points_numba(center_x: float, center_y: float, center_z: float,
                                missile_x: float, missile_y: float, missile_z: float,
                                radius: float, is_top: bool):
        """获取圆上的三个关键点 - Numba优化版本"""
        if is_top:
            # 上层：center到missile的方向
            dx = missile_x - center_x
            dy = missile_y - center_y
        else:
            # 下层：missile到center的方向
            dx = center_x - missile_x
            dy = center_y - missile_y
        
        # 单位化xy投影向量
        xy_norm = sqrt(dx * dx + dy * dy)
        if xy_norm < 1e-10:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / xy_norm, dy / xy_norm
        
        # 垂直向量（逆时针90度）
        vx, vy = -uy, ux
        
        # 三个点的坐标
        points_x = np.array([
            center_x + radius * ux,
            center_x + radius * vx,
            center_x - radius * vx
        ])
        points_y = np.array([
            center_y + radius * uy,
            center_y + radius * vy,
            center_y - radius * vy
        ])
        points_z = np.array([center_z, center_z, center_z])
        
        return points_x, points_y, points_z
else:
    # 标准Python实现作为后备
    def _distance_between_numba(x0, y0, z0, x1, y1, z1):
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        return sqrt(dx * dx + dy * dy + dz * dz)
    
    def _foot_of_perpendicular_numba(px, py, pz, sx, sy, sz, ex, ey, ez):
        seg_x, seg_y, seg_z = ex - sx, ey - sy, ez - sz
        segment_length_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
        
        if segment_length_sq == 0.0:
            return (sx, sy, sz)
        
        to_x, to_y, to_z = px - sx, py - sy, pz - sz
        t = (to_x * seg_x + to_y * seg_y + to_z * seg_z) / segment_length_sq
        t = max(0.0, min(1.0, t))
        
        return (sx + t * seg_x, sy + t * seg_y, sz + t * seg_z)
    
    def _get_circle_points_numba(center_x, center_y, center_z, missile_x, missile_y, missile_z, radius, is_top):
        if is_top:
            dx, dy = missile_x - center_x, missile_y - center_y
        else:
            dx, dy = center_x - missile_x, center_y - missile_y
        
        xy_norm = sqrt(dx * dx + dy * dy)
        if xy_norm < 1e-10:
            ux, uy = 1.0, 0.0
        else:
            ux, uy = dx / xy_norm, dy / xy_norm
        
        vx, vy = -uy, ux
        
        points_x = np.array([center_x + radius * ux, center_x + radius * vx, center_x - radius * vx])
        points_y = np.array([center_y + radius * uy, center_y + radius * vy, center_y - radius * vy])
        points_z = np.array([center_z, center_z, center_z])
        
        return points_x, points_y, points_z

# =============================================================================
# 优化后的公共接口函数
# =============================================================================

@lru_cache(maxsize=1000)
def distance_to_origin(point: Vector3) -> float:
    """计算点到原点 (0,0,0) 的距离"""
    if len(point) != 3:
        raise ValueError("point 必须是长度为 3 的三元组")
    return sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])


def distance_between(p0: Vector3, p1: Vector3) -> float:
    """计算两点间的欧氏距离"""
    if len(p0) != 3 or len(p1) != 3:
        raise ValueError("p0 与 p1 必须是长度为 3 的三元组")
    return _distance_between_numba(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2])


@lru_cache(maxsize=1000)
def distance_point_to_line(point: Vector3, line_point: Vector3, line_direction: Vector3) -> float:
    """
    计算点到直线最短距离
    
    直线点向式: P(t) = line_point + t * line_direction
    距离公式: |(point - line_point) × line_direction| / |line_direction|
    """
    if len(point) != 3 or len(line_point) != 3 or len(line_direction) != 3:
        raise ValueError("所有输入都必须是长度为 3 的三元组")

    vx = line_direction[0]
    vy = line_direction[1]
    vz = line_direction[2]
    v_norm = sqrt(vx * vx + vy * vy + vz * vz)
    if v_norm == 0.0:
        raise ValueError("line_direction 不能为零向量")

    px = point[0] - line_point[0]
    py = point[1] - line_point[1]
    pz = point[2] - line_point[2]

    # 叉积 p × v
    cx = py * vz - pz * vy
    cy = pz * vx - px * vz
    cz = px * vy - py * vx

    c_norm = sqrt(cx * cx + cy * cy + cz * cz)
    return c_norm / v_norm


@lru_cache(maxsize=500)
def subtract(a: Vector3, b: Vector3) -> Vector3:
    """计算向量 a - b"""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@lru_cache(maxsize=500)
def dot(a: Vector3, b: Vector3) -> float:
    """计算向量 a 和 b 的点积"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def foot_of_perpendicular_on_segment(point: Vector3, segment_start: Vector3, segment_end: Vector3) -> Vector3:
    """
    计算点到线段的垂足坐标，确保垂足在线段内部
    
    Args:
        point: 要计算垂足的点
        segment_start: 线段起点
        segment_end: 线段终点
    
    Returns:
        垂足坐标（如果垂足在线段外，则返回线段上最近的点）
    """
    return _foot_of_perpendicular_numba(
        point[0], point[1], point[2],
        segment_start[0], segment_start[1], segment_start[2],
        segment_end[0], segment_end[1], segment_end[2]
    )


def feet_of_perpendicular_to_anchor_target_lines(
    point: Vector3, 
    anchor: Vector3, 
    targets: List[Vector3]
) -> List[Vector3]:
    """
    计算一个点到多条"锚点-目标点"线段的垂足坐标
    
    Args:
        point: 要计算垂足的点
        anchor: 锚点（所有线段的起点）
        targets: 目标点列表（每条线段的终点）
    
    Returns:
        垂足坐标列表
    """
    feet = []
    for target in targets:
        foot = foot_of_perpendicular_on_segment(point, anchor, target)
        feet.append(foot)
    return feet


def get_top_plane_points(traj0_pos: Vector3, center: Vector3, radius: float = 7.0) -> List[Vector3]:
    """在 xy 平面上，以 center 为圆心、radius 为半径的圆上，计算与 traj0-center 连线垂直的三个点（基于cyx实现）"""
    points_x, points_y, points_z = _get_circle_points_numba(
        center[0], center[1], center[2],
        traj0_pos[0], traj0_pos[1], traj0_pos[2],
        radius, True  # is_top = True
    )
    return [(points_x[i], points_y[i], points_z[i]) for i in range(3)]


def get_under_points(traj0_pos: Vector3, center: Vector3, radius: float = 7.0) -> List[Vector3]:
    """在 xy 平面上，以 center 为圆心、radius 为半径的圆上，计算与 traj0-center 连线垂直的三个点（基于cyx实现）"""
    points_x, points_y, points_z = _get_circle_points_numba(
        center[0], center[1], center[2],
        traj0_pos[0], traj0_pos[1], traj0_pos[2],
        radius, False  # is_top = False
    )
    return [(points_x[i], points_y[i], points_z[i]) for i in range(3)]


def get_target_points(missile_pos: Vector3, center: Vector3, radius: float = 7.0) -> List[Vector3]:
    """
    获取所有目标点（兼容上下两层）
    
    Args:
        missile_pos: 导弹当前位置
        center: 圆心位置
        radius: 圆半径
    
    Returns:
        目标点列表（3个点）
    """
    return get_top_plane_points(missile_pos, center, radius)


@lru_cache(maxsize=100)
def refine_boundary(
    predicate: Callable[[float, float], bool],
    t_lo: float,
    t_hi: float,
    threshold: float,
    iters: int = 25,
) -> float:
    """在 [t_lo, t_hi] 上用二分细化边界（predicate 在两端取值不同）"""
    v_lo = predicate(t_lo, threshold)
    v_hi = predicate(t_hi, threshold)
    if v_lo == v_hi:
        return t_lo
    for _ in range(iters):
        mid = 0.5 * (t_lo + t_hi)
        v_mid = predicate(mid, threshold)
        if v_mid == v_lo:
            t_lo = mid
            v_lo = v_mid
        else:
            t_hi = mid
            v_hi = v_mid
    return 0.5 * (t_lo + t_hi)


def find_t_intervals(
    predicate: Callable[[float, float], bool],
    threshold: float,
    t_start: float,
    t_end: float,
    step: float,
) -> List[Tuple[float, float]]:
    """扫描 + 二分：寻找满足 predicate(t, threshold) 的所有 t 区间"""
    intervals: List[Tuple[float, float]] = []
    t = t_start
    prev_t = t
    prev_in = predicate(prev_t, threshold)
    seg_start: float | None = prev_t if prev_in else None

    while t <= t_end:
        cur_in = predicate(t, threshold)
        if prev_in != cur_in:
            boundary = refine_boundary(predicate, prev_t, t, threshold)
            if prev_in and not cur_in:
                if seg_start is not None:
                    intervals.append((seg_start, boundary))
                    seg_start = None
            elif (not prev_in) and cur_in:
                seg_start = boundary
        prev_t = t
        prev_in = cur_in
        t += step

    if prev_in:
        end_boundary = refine_boundary(predicate, max(t_end - step, t_start), t_end, threshold)
        if seg_start is None:
            seg_start = t_start
        intervals.append((seg_start, end_boundary))
    return intervals


@lru_cache(maxsize=360)
def direction_to_unit_vector(direction_degrees: float) -> Tuple[float, float]:
    """
    将角度（度）转换为xy平面单位方向向量
    
    Args:
        direction_degrees: 角度（度），0度为x轴正向，逆时针为正
    
    Returns:
        (ux, uy): xy平面单位方向向量
    """
    rad = radians(direction_degrees)
    return (cos(rad), sin(rad))
