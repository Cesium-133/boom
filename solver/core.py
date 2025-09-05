"""
烟幕遮蔽计算核心模块

实现导弹拦截过程中烟幕弹有效遮蔽时长的核心计算逻辑。
包括单无人机、多无人机、单/多烟幕弹等各种计算场景。
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from .config import CONSTANTS, TARGETS, CALCULATION_PARAMS, SMOKE_PARAMS
from .geometry import (
    Vector3, distance_between, get_top_plane_points, get_under_points,
    feet_of_perpendicular_to_anchor_target_lines, find_t_intervals
)
from .trajectory import TrajectoryCalculator


class MaskingCalculator:
    """遮蔽计算器"""
    
    def __init__(self):
        self.traj_calc = TrajectoryCalculator()
        self.threshold = SMOKE_PARAMS["cloud_radius"]
        self.time_step = CALCULATION_PARAMS["time_step"]
        self.max_time = CALCULATION_PARAMS["max_simulation_time"]
        
        # 目标配置
        self.target_centers = [
            TARGETS["true_target"]["base_center"].copy(),  # 基础中心
            [TARGETS["true_target"]["base_center"][0], 
             TARGETS["true_target"]["base_center"][1], 
             TARGETS["true_target"]["base_center"][2] + TARGETS["true_target"]["height"]]  # 上层中心
        ]
        self.target_radius = TARGETS["true_target"]["radius"]
    
    def _create_masking_predicate(
        self,
        missile_traj: Callable[[float], Vector3],
        smoke_traj: Callable[[float], Vector3]
    ) -> Callable[[float, float], bool]:
        """
        创建遮蔽判定函数
        
        Args:
            missile_traj: 导弹轨迹函数
            smoke_traj: 烟幕云轨迹函数
            
        Returns:
            判定函数 predicate(t, threshold) -> bool
        """
        def predicate(t: float, threshold: float) -> bool:
            # 获取当前时刻的导弹和烟幕云位置
            missile_pos = missile_traj(t)
            smoke_pos = smoke_traj(t)
            
            # 计算所有目标点（基于cyx实现：上下两层各3个点）
            top_points = get_top_plane_points(missile_pos, self.target_centers[1], self.target_radius)  # 上层
            under_points = get_under_points(missile_pos, self.target_centers[0], self.target_radius)   # 下层
            all_target_points = top_points + under_points
            
            # 计算烟幕云到各条视线的距离
            feet = feet_of_perpendicular_to_anchor_target_lines(
                smoke_pos, missile_pos, all_target_points
            )
            
            distances = []
            for foot in feet:
                dist = distance_between(smoke_pos, foot)
                distances.append(dist)
            
            # 判断是否所有距离都小于等于阈值
            return max(distances) <= threshold if distances else False
        
        return predicate
    
    def calculate_masking_duration(
        self,
        missile_traj: Callable[[float], Vector3],
        smoke_traj: Callable[[float], Vector3],
        start_time: float = 0.0,
        end_time: float = None
    ) -> float:
        """
        计算有效遮蔽时长
        
        Args:
            missile_traj: 导弹轨迹函数
            smoke_traj: 烟幕云轨迹函数
            start_time: 开始时间
            end_time: 结束时间（如果为None则自动计算）
            
        Returns:
            有效遮蔽时长（秒）
        """
        if end_time is None:
            # 自动计算合理的结束时间
            _, end_time = self.traj_calc.get_trajectory_bounds(missile_traj, self.max_time)
        
        # 创建遮蔽判定函数
        predicate = self._create_masking_predicate(missile_traj, smoke_traj)
        
        # 寻找满足遮蔽条件的时间区间
        intervals = find_t_intervals(
            predicate, 
            self.threshold, 
            start_time, 
            end_time, 
            self.time_step
        )
        
        # 计算总遮蔽时长
        total_duration = sum(b - a for a, b in intervals)
        return total_duration


def calculate_single_uav_single_smoke_masking(
    uav_direction: float,
    uav_speed: float,
    smoke_deploy_time: float,
    smoke_explode_delay: float
) -> float:
    """
    计算单无人机单烟幕弹单导弹的有效遮蔽时长
    
    Args:
        uav_direction: 无人机飞行方向（度，0-360）
        uav_speed: 无人机飞行速度（m/s）
        smoke_deploy_time: 烟幕弹投放时间（s）
        smoke_explode_delay: 烟幕弹起爆相对延时（s）
        
    Returns:
        有效遮蔽时长（s）
    """
    calc = MaskingCalculator()
    
    # 创建导弹轨迹
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 创建无人机轨迹
    uav_traj = calc.traj_calc.create_uav_trajectory("FY1", direction_degrees=uav_direction, speed=uav_speed)
    
    # 创建烟幕云轨迹
    smoke_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_deploy_time, smoke_explode_delay)
    
    # 计算遮蔽时长
    explode_time = smoke_deploy_time + smoke_explode_delay
    duration = calc.calculate_masking_duration(missile_traj, smoke_traj, start_time=explode_time)
    
    return duration


def calculate_single_uav_triple_smoke_masking(
    uav_direction: float,
    uav_speed: float,
    smoke_a_deploy_time: float,
    smoke_a_explode_delay: float,
    smoke_b_deploy_delay: float,
    smoke_b_explode_delay: float,
    smoke_c_deploy_delay: float,
    smoke_c_explode_delay: float
) -> float:
    """
    计算单无人机3烟幕弹单导弹的有效遮蔽时长
    
    Args:
        uav_direction: 无人机飞行方向（度，0-360）
        uav_speed: 无人机飞行速度（m/s）
        smoke_a_deploy_time: 烟幕弹A投放时间（s）
        smoke_a_explode_delay: 烟幕弹A起爆延时（s）
        smoke_b_deploy_delay: 烟幕弹B相对A的投放延时（s）
        smoke_b_explode_delay: 烟幕弹B起爆延时（s）
        smoke_c_deploy_delay: 烟幕弹C相对B的投放延时（s）
        smoke_c_explode_delay: 烟幕弹C起爆延时（s）
        
    Returns:
        有效遮蔽时长（s）
    """
    # 计算绝对投放时间
    smoke_b_deploy_time = smoke_a_deploy_time + smoke_b_deploy_delay
    smoke_c_deploy_time = smoke_b_deploy_time + smoke_c_deploy_delay
    
    calc = MaskingCalculator()
    
    # 创建导弹轨迹
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 创建无人机轨迹（同一架无人机）
    uav_traj = calc.traj_calc.create_uav_trajectory("FY1", direction_degrees=uav_direction, speed=uav_speed)
    
    # 创建三个烟幕云轨迹
    smoke_a_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_a_deploy_time, smoke_a_explode_delay)
    smoke_b_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_b_deploy_time, smoke_b_explode_delay)
    smoke_c_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_c_deploy_time, smoke_c_explode_delay)
    
    # 创建组合遮蔽判定函数
    def combined_predicate(t: float, threshold: float) -> bool:
        """组合判定：任一烟幕云满足遮蔽条件即可"""
        missile_pos = missile_traj(t)
        
        # 计算所有目标点（基于cyx实现：上下两层各3个点）
        top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)  # 上层
        under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)   # 下层
        all_target_points = top_points + under_points
        
        # 检查每个烟幕云的遮蔽效果
        for smoke_traj in [smoke_a_traj, smoke_b_traj, smoke_c_traj]:
            smoke_pos = smoke_traj(t)
            feet = feet_of_perpendicular_to_anchor_target_lines(
                smoke_pos, missile_pos, all_target_points
            )
            
            distances = [distance_between(smoke_pos, foot) for foot in feet]
            if distances and max(distances) <= threshold:
                return True
        
        return False
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 寻找满足条件的时间区间
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    intervals = find_t_intervals(
        combined_predicate,
        calc.threshold,
        earliest_explode,
        end_time,
        calc.time_step
    )
    
    # 计算总遮蔽时长
    total_duration = sum(b - a for a, b in intervals)
    return total_duration


def calculate_multi_uav_single_smoke_masking(
    uav_a_direction: float,
    uav_a_speed: float,
    uav_b_direction: float,
    uav_b_speed: float,
    uav_c_direction: float,
    uav_c_speed: float,
    smoke_a_deploy_time: float,
    smoke_a_explode_delay: float,
    smoke_b_deploy_time: float,
    smoke_b_explode_delay: float,
    smoke_c_deploy_time: float,
    smoke_c_explode_delay: float
) -> float:
    """
    计算多无人机单烟幕弹单导弹的有效遮蔽时长
    
    Args:
        uav_a_direction: 无人机A飞行方向（度，0-360）
        uav_a_speed: 无人机A飞行速度（m/s）
        uav_b_direction: 无人机B飞行方向（度，0-360）
        uav_b_speed: 无人机B飞行速度（m/s）
        uav_c_direction: 无人机C飞行方向（度，0-360）
        uav_c_speed: 无人机C飞行速度（m/s）
        smoke_a_deploy_time: 烟幕弹A投放时间（s）
        smoke_a_explode_delay: 烟幕弹A起爆延时（s）
        smoke_b_deploy_time: 烟幕弹B投放时间（s）
        smoke_b_explode_delay: 烟幕弹B起爆延时（s）
        smoke_c_deploy_time: 烟幕弹C投放时间（s）
        smoke_c_explode_delay: 烟幕弹C起爆延时（s）
        
    Returns:
        有效遮蔽时长（s）
    """
    calc = MaskingCalculator()
    
    # 创建导弹轨迹
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 创建三架不同无人机的轨迹
    uav_a_traj = calc.traj_calc.create_uav_trajectory("FY1", direction_degrees=uav_a_direction, speed=uav_a_speed)
    uav_b_traj = calc.traj_calc.create_uav_trajectory("FY2", direction_degrees=uav_b_direction, speed=uav_b_speed)
    uav_c_traj = calc.traj_calc.create_uav_trajectory("FY3", direction_degrees=uav_c_direction, speed=uav_c_speed)
    
    # 创建三个烟幕云轨迹（每架无人机投放一个烟幕弹）
    smoke_a_traj = calc.traj_calc.create_smoke_trajectory(uav_a_traj, smoke_a_deploy_time, smoke_a_explode_delay)
    smoke_b_traj = calc.traj_calc.create_smoke_trajectory(uav_b_traj, smoke_b_deploy_time, smoke_b_explode_delay)
    smoke_c_traj = calc.traj_calc.create_smoke_trajectory(uav_c_traj, smoke_c_deploy_time, smoke_c_explode_delay)
    
    # 创建组合遮蔽判定函数
    def combined_predicate(t: float, threshold: float) -> bool:
        """组合判定：任一烟幕云满足遮蔽条件即可"""
        missile_pos = missile_traj(t)
        
        # 计算所有目标点（基于cyx实现：上下两层各3个点）
        top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)  # 上层
        under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)   # 下层
        all_target_points = top_points + under_points
        
        # 检查每个烟幕云的遮蔽效果
        for smoke_traj in [smoke_a_traj, smoke_b_traj, smoke_c_traj]:
            smoke_pos = smoke_traj(t)
            feet = feet_of_perpendicular_to_anchor_target_lines(
                smoke_pos, missile_pos, all_target_points
            )
            
            distances = [distance_between(smoke_pos, foot) for foot in feet]
            if distances and max(distances) <= threshold:
                return True
        
        return False
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 寻找满足条件的时间区间
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    intervals = find_t_intervals(
        combined_predicate,
        calc.threshold,
        earliest_explode,
        end_time,
        calc.time_step
    )
    
    # 计算总遮蔽时长
    total_duration = sum(b - a for a, b in intervals)
    return total_duration


def calculate_single_uav_triple_smoke_masking_multiple(
    uav_direction: float,
    uav_speed: float,
    smoke_a_deploy_time: float,
    smoke_a_explode_delay: float,
    smoke_b_deploy_delay: float,
    smoke_b_explode_delay: float,
    smoke_c_deploy_delay: float,
    smoke_c_explode_delay: float
) -> float:
    """
    计算单无人机3烟幕弹单导弹的有效遮蔽时长 - 联合遮挡版本
    
    与 calculate_single_uav_triple_smoke_masking 的区别：
    - 原版本：任意一个烟幕弹独立满足遮挡条件即可
    - 此版本：考虑多个烟幕弹联合遮挡的协同效应
    
    Args:
        uav_direction: 无人机飞行方向（度，0-360）
        uav_speed: 无人机飞行速度（m/s）
        smoke_a_deploy_time: 烟幕弹A投放时间（s）
        smoke_a_explode_delay: 烟幕弹A起爆延时（s）
        smoke_b_deploy_delay: 烟幕弹B相对A的投放延时（s）
        smoke_b_explode_delay: 烟幕弹B起爆延时（s）
        smoke_c_deploy_delay: 烟幕弹C相对B的投放延时（s）
        smoke_c_explode_delay: 烟幕弹C起爆延时（s）
        
    Returns:
        有效遮蔽时长（s）
    """
    # 导入必要的函数（从 for_merge.py 适配）
    from .for_merge import (
        is_cylinder_covered_at_t, 
        calculate_tangent_plane,
        compute_tangent_cone_to_sphere,
        project_point_to_plane
    )
    import numpy as np
    
    # 计算绝对投放时间
    smoke_b_deploy_time = smoke_a_deploy_time + smoke_b_deploy_delay
    smoke_c_deploy_time = smoke_b_deploy_time + smoke_c_deploy_delay
    
    calc = MaskingCalculator()
    
    # 创建导弹轨迹
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 创建无人机轨迹（同一架无人机）
    uav_traj = calc.traj_calc.create_uav_trajectory("FY1", direction_degrees=uav_direction, speed=uav_speed)
    
    # 创建三个烟幕云轨迹
    smoke_a_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_a_deploy_time, smoke_a_explode_delay)
    smoke_b_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_b_deploy_time, smoke_b_explode_delay)
    smoke_c_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_c_deploy_time, smoke_c_explode_delay)
    
    # 创建联合遮蔽判定函数
    def multiple_smoke_predicate(t: float, threshold: float) -> bool:
        """
        联合判定：多个烟幕弹的联合投影是否完全覆盖目标
        
        核心思想：
        1. 将每个烟幕弹建模为一个球体（半径=threshold）
        2. 从导弹位置向每个球体发射切锥
        3. 计算所有切锥在目标切平面上的投影
        4. 判断这些投影的并集是否完全覆盖目标的投影
        """
        missile_pos = missile_traj(t)
        
        # 获取当前时刻所有活跃的烟幕弹位置
        active_smoke_positions = []
        
        # 检查烟幕弹A是否活跃
        if t >= smoke_a_deploy_time + smoke_a_explode_delay:
            smoke_a_pos = smoke_a_traj(t)
            active_smoke_positions.append(smoke_a_pos)
        
        # 检查烟幕弹B是否活跃
        if t >= smoke_b_deploy_time + smoke_b_explode_delay:
            smoke_b_pos = smoke_b_traj(t)
            active_smoke_positions.append(smoke_b_pos)
        
        # 检查烟幕弹C是否活跃
        if t >= smoke_c_deploy_time + smoke_c_explode_delay:
            smoke_c_pos = smoke_c_traj(t)
            active_smoke_positions.append(smoke_c_pos)
        
        # 如果没有活跃的烟幕弹，返回False
        if not active_smoke_positions:
            return False
        
        # 如果只有一个活跃的烟幕弹，回退到原始的单独判定逻辑
        if len(active_smoke_positions) == 1:
            smoke_pos = active_smoke_positions[0]
            top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)
            under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)
            all_target_points = top_points + under_points
            
            feet = feet_of_perpendicular_to_anchor_target_lines(
                smoke_pos, missile_pos, all_target_points
            )
            distances = [distance_between(smoke_pos, foot) for foot in feet]
            return distances and max(distances) <= threshold
        
        # 多烟幕弹联合遮挡判定
        # 使用 for_merge.py 中的逻辑
        def missile_traj_wrapper(time):
            return missile_pos  # 在当前时刻，导弹位置固定
        
        def smoke_traj_wrapper(time):
            return active_smoke_positions  # 返回所有活跃烟幕弹位置
        
        # 目标几何参数
        center0 = tuple(calc.target_centers[0])  # 底部中心
        center1 = tuple(calc.target_centers[1])  # 顶部中心
        cylinder_radius = calc.target_radius
        sphere_radius_for_cone = threshold  # 烟幕云半径
        
        # 调用联合遮挡判定函数
        return is_cylinder_covered_at_t(
            t, missile_traj_wrapper, smoke_traj_wrapper, 
            center0, center1, cylinder_radius, sphere_radius_for_cone
        )
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 寻找满足条件的时间区间
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    intervals = find_t_intervals(
        multiple_smoke_predicate,
        calc.threshold,
        earliest_explode,
        end_time,
        calc.time_step
    )
    
    # 计算总遮蔽时长
    total_duration = sum(b - a for a, b in intervals)
    return total_duration


# 所有计算函数已在上面定义完成
