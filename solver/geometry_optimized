"""
几何计算工具模块 - Numba 优化版本

使用 Numba JIT 编译器加速核心数值计算
性能提升预期：5-50倍
"""

import numpy as np
from numba import jit, types
from typing import List, Tuple, Callable
from math import sqrt, cos, sin, radians

Vector3 = Tuple[float, float, float]

# =============================================================================
# 核心计算函数 - Numba 优化版本
# =============================================================================

@jit(nopython=True, cache=True, fastmath=True)
def distance_between_numba(x0: float, y0: float, z0: float, 
                          x1: float, y1: float, z1: float) -> float:
    """计算两点间的欧氏距离 - Numba 优化版本"""
    dx = x1 - x0
    dy = y1 - y0
    dz = z1 - z0
    return sqrt(dx * dx + dy * dy + dz * dz)

@jit(nopython=True, cache=True, fastmath=True)
def distance_point_to_line_numba(px: float, py: float, pz: float,
                                lx: float, ly: float, lz: float,
                                vx: float, vy: float, vz: float) -> float:
    """计算点到直线最短距离 - Numba 优化版本"""
    v_norm_sq = vx * vx + vy * vy + vz * vz
    if v_norm_sq == 0.0:
        return 0.0
    
    dx = px - lx
    dy = py - ly
    dz = pz - lz
    
    # 叉积 d × v
    cx = dy * vz - dz * vy
    cy = dz * vx - dx * vz
    cz = dx * vy - dy * vx
    
    c_norm_sq = cx * cx + cy * cy + cz * cz
    return sqrt(c_norm_sq / v_norm_sq)

@jit(nopython=True, cache=True, fastmath=True)
def foot_of_perpendicular_on_segment_numba(px: float, py: float, pz: float,
                                          sx: float, sy: float, sz: float,
                                          ex: float, ey: float, ez: float):
    """计算点到线段的垂足坐标 - Numba 优化版本"""
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
def compute_masking_distances_numba(smoke_x: float, smoke_y: float, smoke_z: float,
                                   missile_x: float, missile_y: float, missile_z: float,
                                   target_points_x: np.ndarray, 
                                   target_points_y: np.ndarray, 
                                   target_points_z: np.ndarray) -> float:
    """
    计算烟幕云到所有视线的最大距离 - Numba 优化版本
    这是最关键的性能瓶颈函数
    """
    max_distance = 0.0
    n_points = len(target_points_x)
    
    for i in range(n_points):
        # 计算垂足
        foot_x, foot_y, foot_z = foot_of_perpendicular_on_segment_numba(
            smoke_x, smoke_y, smoke_z,
            missile_x, missile_y, missile_z,
            target_points_x[i], target_points_y[i], target_points_z[i]
        )
        
        # 计算距离
        dist = distance_between_numba(smoke_x, smoke_y, smoke_z, foot_x, foot_y, foot_z)
        if dist > max_distance:
            max_distance = dist
    
    return max_distance

@jit(nopython=True, cache=True, fastmath=True)
def get_circle_points_numba(center_x: float, center_y: float, center_z: float,
                           missile_x: float, missile_y: float, missile_z: float,
                           radius: float):
    """获取圆上的关键点 - Numba 优化版本"""
    # 从圆心到导弹的方向向量
    dx = missile_x - center_x
    dy = missile_y - center_y
    dz = missile_z - center_z
    
    d_norm = sqrt(dx * dx + dy * dy + dz * dz)
    if d_norm == 0.0:
        # 特殊情况：导弹在圆心正上方
        return np.array([center_x + radius, center_x - radius/2, center_x - radius/2]), \
               np.array([center_y, center_y + radius*0.866, center_y - radius*0.866]), \
               np.array([center_z, center_z, center_z])
    
    dx_norm = dx / d_norm
    dy_norm = dy / d_norm
    dz_norm = dz / d_norm
    
    # 构建垂直向量
    if abs(dx_norm) < 0.9:
        aux_x, aux_y, aux_z = 1.0, 0.0, 0.0
    else:
        aux_x, aux_y, aux_z = 0.0, 1.0, 0.0
    
    # 第一个垂直向量
    v1_x = dy_norm * aux_z - dz_norm * aux_y
    v1_y = dz_norm * aux_x - dx_norm * aux_z
    v1_z = dx_norm * aux_y - dy_norm * aux_x
    
    v1_norm = sqrt(v1_x * v1_x + v1_y * v1_y + v1_z * v1_z)
    v1_x /= v1_norm
    v1_y /= v1_norm
    v1_z /= v1_norm
    
    # 第二个垂直向量
    v2_x = dy_norm * v1_z - dz_norm * v1_y
    v2_y = dz_norm * v1_x - dx_norm * v1_z
    v2_z = dx_norm * v1_y - dy_norm * v1_x
    
    # 生成三个点（120度间隔）
    angles = np.array([0.0, 2.0944, 4.1888])  # 0, 2π/3, 4π/3
    
    points_x = np.zeros(3)
    points_y = np.zeros(3)
    points_z = np.zeros(3)
    
    for i in range(3):
        cos_angle = cos(angles[i])
        sin_angle = sin(angles[i])
        
        points_x[i] = center_x + radius * (cos_angle * v1_x + sin_angle * v2_x)
        points_y[i] = center_y + radius * (cos_angle * v1_y + sin_angle * v2_y)
        points_z[i] = center_z + radius * (cos_angle * v1_z + sin_angle * v2_z)
    
    return points_x, points_y, points_z

# =============================================================================
# 兼容性包装函数
# =============================================================================

def distance_between(p0: Vector3, p1: Vector3) -> float:
    """计算两点间的欧氏距离 - 兼容性包装"""
    return distance_between_numba(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2])

def distance_point_to_line(point: Vector3, line_point: Vector3, line_direction: Vector3) -> float:
    """计算点到直线最短距离 - 兼容性包装"""
    return distance_point_to_line_numba(
        point[0], point[1], point[2],
        line_point[0], line_point[1], line_point[2],
        line_direction[0], line_direction[1], line_direction[2]
    )

def get_top_plane_points(missile_pos: Vector3, top_center: Vector3, radius: float) -> List[Vector3]:
    """获取上层圆平面的三个关键点 - 兼容性包装"""
    points_x, points_y, points_z = get_circle_points_numba(
        top_center[0], top_center[1], top_center[2],
        missile_pos[0], missile_pos[1], missile_pos[2],
        radius
    )
    return [(points_x[i], points_y[i], points_z[i]) for i in range(3)]

# 导入其他非性能关键的函数
from .geometry import (
    get_under_points,
    feet_of_perpendicular_to_anchor_target_lines,
    find_t_intervals,
    direction_to_unit_vector,
    subtract,
    dot
)