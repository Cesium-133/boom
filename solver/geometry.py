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

Vector3 = Tuple[float, float, float]


def distance_to_origin(point: Vector3) -> float:
    """计算点到原点 (0,0,0) 的距离"""
    if len(point) != 3:
        raise ValueError("point 必须是长度为 3 的三元组")
    return sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2])


def distance_between(p0: Vector3, p1: Vector3) -> float:
    """计算两点间的欧氏距离"""
    if len(p0) != 3 or len(p1) != 3:
        raise ValueError("p0 与 p1 必须是长度为 3 的三元组")
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1] 
    dz = p1[2] - p0[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


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


def subtract(a: Vector3, b: Vector3) -> Vector3:
    """计算向量 a - b"""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


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
    # 线段方向向量
    segment_vec = subtract(segment_end, segment_start)
    segment_length_sq = dot(segment_vec, segment_vec)
    
    # 如果线段长度为0，返回起点
    if segment_length_sq == 0:
        return segment_start
    
    # 从线段起点到目标点的向量
    to_point = subtract(point, segment_start)
    
    # 计算投影参数 t
    t = dot(to_point, segment_vec) / segment_length_sq
    
    # 将 t 限制在 [0, 1] 范围内，确保垂足在线段上
    t = max(0.0, min(1.0, t))
    
    # 计算垂足坐标
    foot = (
        segment_start[0] + t * segment_vec[0],
        segment_start[1] + t * segment_vec[1],
        segment_start[2] + t * segment_vec[2]
    )
    
    return foot


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
    # center 到 traj0 的向量在 xy 平面的投影
    dx = - center[0] + traj0_pos[0]
    dy = - center[1] + traj0_pos[1]
    
    # 单位化 xy 投影向量
    xy_norm = (dx * dx + dy * dy) ** 0.5
    if xy_norm < 1e-10:
        # 如果 traj0 和 center 在 xy 平面重合，使用默认方向
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / xy_norm, dy / xy_norm
    
    # 垂直向量（逆时针90度）
    vx, vy = -uy, ux
    
    # 三个点：沿连线方向、垂直方向、-垂直方向
    points = []
    # 点1：沿连线方向（从 center 指向 traj0 的方向）
    px1 = center[0] + radius * ux
    py1 = center[1] + radius * uy
    points.append((px1, py1, center[2]))
    
    # 点2：沿垂直方向
    px2 = center[0] + radius * vx
    py2 = center[1] + radius * vy
    points.append((px2, py2, center[2]))
    
    # 点3：沿-垂直方向
    px3 = center[0] - radius * vx
    py3 = center[1] - radius * vy
    points.append((px3, py3, center[2]))
    
    return points


def get_under_points(traj0_pos: Vector3, center: Vector3, radius: float = 7.0) -> List[Vector3]:
    """在 xy 平面上，以 center 为圆心、radius 为半径的圆上，计算与 traj0-center 连线垂直的三个点（基于cyx实现）"""
    # center 到 traj0 的向量在 xy 平面的投影
    dx = center[0] - traj0_pos[0]
    dy = center[1] - traj0_pos[1]
    
    # 单位化 xy 投影向量
    xy_norm = (dx * dx + dy * dy) ** 0.5
    if xy_norm < 1e-10:
        # 如果 traj0 和 center 在 xy 平面重合，使用默认方向
        ux, uy = 1.0, 0.0
    else:
        ux, uy = dx / xy_norm, dy / xy_norm
    
    # 垂直向量（逆时针90度）
    vx, vy = -uy, ux
    
    # 三个点：沿连线方向、垂直方向、-垂直方向
    points = []
    # 点1：沿连线方向（从 center 指向 traj0 的方向）
    px1 = center[0] + radius * ux
    py1 = center[1] + radius * uy
    points.append((px1, py1, center[2]))
    
    # 点2：沿垂直方向
    px2 = center[0] + radius * vx
    py2 = center[1] + radius * vy
    points.append((px2, py2, center[2]))
    
    # 点3：沿-垂直方向
    px3 = center[0] - radius * vx
    py3 = center[1] - radius * vy
    points.append((px3, py3, center[2]))
    
    return points


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
