from __future__ import annotations

from typing import Tuple, Callable, List
from .geometry import find_t_intervals  # reuse interval finder
import numpy as np

Vector3 = Tuple[float, float, float]


def project_point_to_plane(point: Vector3, plane_normal: Vector3, plane_d: float) -> Vector3:
    """
    将点投影到平面上。
    平面方程：plane_normal·x + plane_d = 0
    """
    # 计算点到平面的距离
    distance = plane_normal[0]*point[0] + plane_normal[1]*point[1] + plane_normal[2]*point[2] + plane_d
    
    # 投影点 = 原点到平面的最近点
    proj_point = (point[0] - distance*plane_normal[0],
                  point[1] - distance*plane_normal[1],
                  point[2] - distance*plane_normal[2])
    return proj_point


def calculate_tangent_plane(center0: Vector3, cylinder_radius: float, traj0_pos: Vector3) -> Tuple[Vector3, float]:
    """
    计算题目三的切平面（法向量沿 center0→traj0 在 xoy 的方向；切点为 xoy 上 center0 + R·dir）。
    返回 (normal, d)，平面方程为: normal·x + d = 0。
    """
    traj0_projection = (traj0_pos[0], traj0_pos[1], 0.0)
    direction_vector = (
        traj0_projection[0] - center0[0],
        traj0_projection[1] - center0[1],
        0.0,
    )
    norm_xy = (direction_vector[0] ** 2 + direction_vector[1] ** 2) ** 0.5
    if norm_xy > 1e-10:
        dir_unit = (direction_vector[0] / norm_xy, direction_vector[1] / norm_xy, 0.0)
    else:
        dir_unit = (1.0, 0.0, 0.0)
    plane_normal = (dir_unit[0], dir_unit[1], 0.0)
    tangent_point = (
        center0[0] + cylinder_radius * dir_unit[0],
        center0[1] + cylinder_radius * dir_unit[1],
        center0[2],
    )
    d = -(
        plane_normal[0] * tangent_point[0]
        + plane_normal[1] * tangent_point[1]
        + plane_normal[2] * tangent_point[2]
    )
    return plane_normal, d


# 已由 calculate_tangent_plane 提供 (n, d)，不再额外导出 abcd 版本


def compute_tangent_cone_to_sphere(apex: Vector3, sphere_center: Vector3, sphere_radius: float) -> Tuple[Vector3, float]:
    """
    计算以 apex 为顶点、与以 sphere_center 为圆心、半径 sphere_radius 的球相切的圆锥（直圆锥）参数。
    - 圆锥轴向量取 apex→sphere_center 方向（单位）
    - 半顶角 alpha 满足: sin(alpha) = R / d, 其中 d 为 apex 与 sphere_center 的距离

    返回: (axis_unit, alpha)
    """
    ax = sphere_center[0] - apex[0]
    ay = sphere_center[1] - apex[1]
    az = sphere_center[2] - apex[2]
    d = (ax*ax + ay*ay + az*az) ** 0.5
    if d < 1e-10 or sphere_radius <= 0.0 or sphere_radius >= d:
        # 退化情形：返回零角并给个默认轴
        return (1.0, 0.0, 0.0), 0.0
    axis_unit = (ax/d, ay/d, az/d)
    # 半顶角
    import math
    alpha = math.asin(sphere_radius / d)
    return axis_unit, alpha


def cone_projected_conic_on_plane(
    plane_abcd: Tuple[float, float, float, float],
    apex: Vector3,
    axis_unit: Vector3,
    alpha: float,
) -> Tuple[Vector3, Vector3, Vector3, Tuple[float, float, float, float, float, float]]:
    """
    计算圆锥在给定平面上的曲线方程（在平面自建坐标系下的二次型系数）。

    圆锥隐式方程（以 apex 为顶点，轴方向 w，半顶角 alpha）:
        ((X-A)·w)^2 = cos^2(alpha) * ||X-A||^2

    在平面上取正交基 (u, v)，平面上一点 O 为 apex 在平面上的正交投影。
    平面内点 X 表示为: X = O + s u + t v。将其代入上式得到关于 (s,t) 的二次型：
        M11 s^2 + 2 M12 s t + M22 t^2 + 2 B1 s + 2 B2 t + C = 0

    返回: (O, u, v, (M11, M22, M12, B1, B2, C))
    """
    import math
    a, b, c, d = plane_abcd
    # 确保法向量单位化
    n_norm = (a*a + b*b + c*c) ** 0.5
    if n_norm < 1e-12:
        raise ValueError("平面法向量长度为零")
    nx, ny, nz = a/n_norm, b/n_norm, c/n_norm

    # 平面上一点 O = A - ((n·A + d)/||n||^2) * n （若已单位化则直接减 (n·A + d) * n）
    n_dot_A = nx*apex[0] + ny*apex[1] + nz*apex[2]
    lam = (n_dot_A + d)  # 因为 n 已单位化
    O = (apex[0] - lam*nx, apex[1] - lam*ny, apex[2] - lam*nz)

    # 在平面内选取正交基 u, v
    # 选一个不平行 n 的向量做初始 t，常用 z 轴或 x 轴
    if abs(nx) < 0.9:
        t0 = (1.0, 0.0, 0.0)
    else:
        t0 = (0.0, 1.0, 0.0)
    # u = normalize(t0 × n)
    ux = t0[1]*nz - t0[2]*ny
    uy = t0[2]*nx - t0[0]*nz
    uz = t0[0]*ny - t0[1]*nx
    u_norm = (ux*ux + uy*uy + uz*uz) ** 0.5
    if u_norm < 1e-12:
        # fallback
        t0 = (0.0, 0.0, 1.0)
        ux = t0[1]*nz - t0[2]*ny
        uy = t0[2]*nx - t0[0]*nz
        uz = t0[0]*ny - t0[1]*nx
        u_norm = (ux*ux + uy*uy + uz*uz) ** 0.5
    ux, uy, uz = ux/u_norm, uy/u_norm, uz/u_norm
    # v = n × u
    vx = ny*uz - nz*uy
    vy = nz*ux - nx*uz
    vz = nx*uy - ny*ux

    # 将轴向量及角度代入二次型
    wx, wy, wz = axis_unit
    cos2 = math.cos(alpha)**2

    # 预计算：向量 O-A = O - apex
    r0 = (O[0] - apex[0], O[1] - apex[1], O[2] - apex[2])

    # 函数用于计算任意线性组合 r = r0 + s u + t v 的项
    def dot_w(x, y, z):
        return wx*x + wy*y + wz*z
    def dot_self(x, y, z):
        return x*x + y*y + z*z

    # 展开 ((r·w)^2 - cos2 * (r·r)) = 0，r = r0 + s u + t v
    # 记：Aw = r0·w， Bw_s = u·w， Bw_t = v·w
    Aw = dot_w(*r0)
    Bw_s = dot_w(ux, uy, uz)
    Bw_t = dot_w(vx, vy, vz)

    # ||r||^2 = r0·r0 + 2 s (r0·u) + 2 t (r0·v) + s^2 (u·u) + 2 s t (u·v) + t^2 (v·v)
    r0_u = r0[0]*ux + r0[1]*uy + r0[2]*uz
    r0_v = r0[0]*vx + r0[1]*vy + r0[2]*vz
    u_u = ux*ux + uy*uy + uz*uz  # =1
    v_v = vx*vx + vy*vy + vz*vz  # =1
    u_v = ux*vx + uy*vy + uz*vz  # =0

    # (r·w)^2 = (Aw + s Bw_s + t Bw_t)^2 = Aw^2 + 2 Aw Bw_s s + 2 Aw Bw_t t + Bw_s^2 s^2 + 2 Bw_s Bw_t s t + Bw_t^2 t^2
    # cos2 * ||r||^2 = cos2 * [ r0r0 + 2 r0_u s + 2 r0_v t + s^2 + 2 (u·v) s t + t^2 ]，但 (u·v)=0

    M11 = Bw_s*Bw_s - cos2 * u_u
    M22 = Bw_t*Bw_t - cos2 * v_v
    M12 = Bw_s*Bw_t - cos2 * u_v  # u_v=0

    B1 = Aw*Bw_s - cos2 * r0_u
    B2 = Aw*Bw_t - cos2 * r0_v

    C = Aw*Aw - cos2 * (r0[0]*r0[0] + r0[1]*r0[1] + r0[2]*r0[2])

    return O, (ux, uy, uz), (vx, vy, vz), (M11, M22, M12, B1, B2, C)




def is_cylinder_covered_at_t(
    t: float,
    traj0_fn: Callable[[float], Vector3],
    traj1_fn: Callable[[float], List[Vector3]],  # 修改为返回多个轨迹点
    center0: Vector3,
    center1: Vector3,
    cylinder_radius: float,
    sphere_radius_for_cone: float = 10.0,
) -> bool:
    """
    判断在时刻 t：多个圆锥投影的并集是否完全覆盖圆柱投影。
    
    方法：
    - 获取 traj0(t) 作为所有圆锥的顶点
    - 获取 traj1(t) 返回的所有轨迹点，每个点对应一个圆锥
    - 计算每个圆锥在切平面上的投影区域
    - 计算这些投影区域的并集
    - 检查并集是否完全覆盖圆柱在切平面上的投影
    """
    import math
    p0_t = traj0_fn(t)
    traj1_points = traj1_fn(t)  # 获取所有轨迹点
    
    # 如果 traj0 位于任何一个轨迹点的球体内，直接返回 True
    for p1_t in traj1_points:
        dx = p0_t[0] - p1_t[0]
        dy = p0_t[1] - p1_t[1]
        dz = p0_t[2] - p1_t[2]
        if (dx*dx + dy*dy + dz*dz) <= (sphere_radius_for_cone * sphere_radius_for_cone):
            return True

    # 计算切平面
    normal, d_val = calculate_tangent_plane(center0, cylinder_radius, p0_t)
    a, b, c = normal
    n_norm = (a*a + b*b + c*c) ** 0.5
    if n_norm < 1e-12:
        return False
    nx, ny, nz = a/n_norm, b/n_norm, c/n_norm

    # 平面上一点 O = A - (n·A + d) n (n 已单位化)
    n_dot_A = nx*p0_t[0] + ny*p0_t[1] + nz*p0_t[2]
    lam = (n_dot_A + d_val)
    O = (p0_t[0] - lam*nx, p0_t[1] - lam*ny, p0_t[2] - lam*nz)

    # 平面正交基 (u, v)
    if abs(nx) < 0.9:
        t0 = (1.0, 0.0, 0.0)
    else:
        t0 = (0.0, 1.0, 0.0)
    ux = t0[1]*nz - t0[2]*ny
    uy = t0[2]*nx - t0[0]*nz
    uz = t0[0]*ny - t0[1]*nx
    u_norm = (ux*ux + uy*uy + uz*uz) ** 0.5
    if u_norm < 1e-12:
        return False
    ux, uy, uz = ux/u_norm, uy/u_norm, uz/u_norm
    vx = ny*uz - nz*uy
    vy = nz*ux - nx*uz
    vz = nx*uy - ny*ux

    # 圆锥参数
    axis_unit, alpha = compute_tangent_cone_to_sphere(p0_t, p1_t, sphere_radius_for_cone)
    wx, wy, wz = axis_unit
    cos2 = math.cos(alpha)**2

    # 预计算量
    r0 = (O[0]-p0_t[0], O[1]-p0_t[1], O[2]-p0_t[2])
    def dot_w(x,y,z):
        return wx*x + wy*y + wz*z
    def quad_F(s, t):
        # r = r0 + s u + t v
        rx = r0[0] + s*ux + t*vx
        ry = r0[1] + s*uy + t*vy
        rz = r0[2] + s*uz + t*vz
        rw = dot_w(rx, ry, rz)
        rr = rx*rx + ry*ry + rz*rz
        return (rw*rw - cos2*rr), rw

    # 从 traj0 视角计算圆柱在切平面上的投影区域（使用新的近似方法）
    # 注意：这个投影区域会随着 traj0(t) 的变化而变化
    # 圆柱轴方向
    axis_vec = (center1[0] - center0[0], center1[1] - center0[1], center1[2] - center0[2])
    axis_len = (axis_vec[0]**2 + axis_vec[1]**2 + axis_vec[2]**2)**0.5
    if axis_len > 1e-10:
        axis_unit = (axis_vec[0]/axis_len, axis_vec[1]/axis_len, axis_vec[2]/axis_len)
    else:
        axis_unit = (0, 0, 1)
    
    # 圆柱轴在切平面上的投影
    top_center = (center0[0], center0[1], center0[2])
    bot_center = (center1[0], center1[1], center1[2])
    axis_proj_start = project_point_to_plane(top_center, normal, d_val)
    axis_proj_end = project_point_to_plane(bot_center, normal, d_val)
    
    # 计算投影宽度
    axis_normal_dot = abs(axis_unit[0]*normal[0] + axis_unit[1]*normal[1] + axis_unit[2]*normal[2])
    proj_width = cylinder_radius * (1 - axis_normal_dot**2)**0.5
    
    # 计算投影区域
    proj_axis = (axis_proj_end[0] - axis_proj_start[0], 
                axis_proj_end[1] - axis_proj_start[1], 
                axis_proj_end[2] - axis_proj_start[2])
    proj_axis_len = (proj_axis[0]**2 + proj_axis[1]**2 + proj_axis[2]**2)**0.5
    if proj_axis_len > 1e-10:
        proj_axis_unit = (proj_axis[0]/proj_axis_len, proj_axis[1]/proj_axis_len, proj_axis[2]/proj_axis_len)
    else:
        proj_axis_unit = (1, 0, 0)
    
    # 垂直于投影轴的方向
    if abs(proj_axis_unit[0]) < 0.9:
        perp_base = (1, 0, 0)
    else:
        perp_base = (0, 1, 0)
    perp_dot = perp_base[0]*proj_axis_unit[0] + perp_base[1]*proj_axis_unit[1] + perp_base[2]*proj_axis_unit[2]
    perp_vec = (perp_base[0] - perp_dot*proj_axis_unit[0],
               perp_base[1] - perp_dot*proj_axis_unit[1],
               perp_base[2] - perp_dot*proj_axis_unit[2])
    perp_len = (perp_vec[0]**2 + perp_vec[1]**2 + perp_vec[2]**2)**0.5
    if perp_len > 1e-10:
        perp_unit = (perp_vec[0]/perp_len, perp_vec[1]/perp_len, perp_vec[2]/perp_len)
    else:
        perp_unit = (0, 1, 0)
    
    # 预计算所有圆锥的参数，避免在循环中重复计算
    cone_params = []
    for p1_t in traj1_points:
        axis_unit_cone, alpha = compute_tangent_cone_to_sphere(p0_t, p1_t, sphere_radius_for_cone)
        wx, wy, wz = axis_unit_cone
        cos2 = math.cos(alpha)**2
        cone_params.append((wx, wy, wz, cos2))
    
    # 预计算量（对所有圆锥都相同）
    r0 = (O[0]-p0_t[0], O[1]-p0_t[1], O[2]-p0_t[2])
    
    # 在投影区域中采样点，检查是否被多个圆锥的并集覆盖
    axis_samples = np.linspace(0, proj_axis_len, 20)
    perp_samples = np.linspace(-proj_width, proj_width, 10)
    
    for axis_t in axis_samples:
        for perp_t in perp_samples:
            # 投影区域中的点
            proj_point = (axis_proj_start[0] + axis_t*proj_axis_unit[0] + perp_t*perp_unit[0],
                         axis_proj_start[1] + axis_t*proj_axis_unit[1] + perp_t*perp_unit[1],
                         axis_proj_start[2] + axis_t*proj_axis_unit[2] + perp_t*perp_unit[2])
            
            # 将投影点转换为平面内坐标 (s,t)
            rx = proj_point[0] - O[0]
            ry = proj_point[1] - O[1]
            rz = proj_point[2] - O[2]
            s = rx*ux + ry*uy + rz*uz
            tt = rx*vx + ry*vy + rz*vz
            
            # 检查这个点是否被任何一个圆锥覆盖
            covered_by_any_cone = False
            for wx, wy, wz, cos2 in cone_params:
                def dot_w(x,y,z):
                    return wx*x + wy*y + wz*z
                def quad_F(s, t):
                    # r = r0 + s u + t v
                    rx = r0[0] + s*ux + t*vx
                    ry = r0[1] + s*uy + t*vy
                    rz = r0[2] + s*uz + t*vz
                    rw = dot_w(rx, ry, rz)
                    rr = rx*rx + ry*ry + rz*rz
                    return (rw*rw - cos2*rr), rw
                
                F_val, rw = quad_F(s, tt)
                
                # 如果被当前圆锥覆盖，标记为已覆盖
                if rw >= 0 and F_val >= -1e-12:
                    covered_by_any_cone = True
                    break
            
            # 如果这个点没有被任何圆锥覆盖，返回 False
            if not covered_by_any_cone:
                return False
    
    return True


def find_full_cover_intervals(
    traj0_fn: Callable[[float], Vector3],
    traj1_fn: Callable[[float], List[Vector3]],  # 修改为返回多个轨迹点
    t_min: float,
    t_max: float,
    step: float,
    sphere_radius_for_cone: float = 10.0,
):
    """
    扫描时间区间，返回圆锥投影完全覆盖圆柱投影的所有时间子区间列表 [(t_start, t_end), ...]
    """
    # 圆柱几何参数（常量）
    center0: Vector3 = (0, 200, 10)  # 中心点
    center1: Vector3 = (0, 200, 0)   # 中心点
    cylinder_radius: float = 7.0
    
    def predicate(t: float, _threshold: float) -> bool:
        return is_cylinder_covered_at_t(
            t, traj0_fn, traj1_fn, center0, center1, cylinder_radius, sphere_radius_for_cone
        )
    # 复用 find_t_intervals 的接口（threshold 无实际用途）
    return find_t_intervals(predicate, 0.0, t_min, t_max, step)


