"""
烟幕遮蔽计算核心模块

实现导弹拦截过程中烟幕弹有效遮蔽时长的核心计算逻辑。
包括单无人机、多无人机、单/多烟幕弹等各种计算场景。
"""

from __future__ import annotations
from typing import List, Tuple, Callable
from functools import lru_cache
import numpy as np

# 尝试导入Numba，如果不可用则使用标准实现
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

from .config import CONSTANTS, TARGETS, CALCULATION_PARAMS, SMOKE_PARAMS
from .geometry import (
    Vector3, distance_between, get_top_plane_points, get_under_points,
    feet_of_perpendicular_to_anchor_target_lines, find_t_intervals,
    find_t_intervals_adaptive, find_t_intervals_smart
)
from .trajectory import TrajectoryCalculator

# =============================================================================
# Numba优化的核心计算函数
# =============================================================================

if HAS_NUMBA:
    @jit(nopython=True, cache=True, fastmath=True)
    def _compute_max_distance_to_lines_numba(smoke_x: float, smoke_y: float, smoke_z: float,
                                           missile_x: float, missile_y: float, missile_z: float,
                                           target_points_x: np.ndarray, 
                                           target_points_y: np.ndarray, 
                                           target_points_z: np.ndarray) -> float:
        """
        计算烟幕云到所有视线的最大距离 - Numba优化版本
        这是性能瓶颈函数，需要重点优化
        """
        max_distance = 0.0
        n_points = len(target_points_x)
        
        for i in range(n_points):
            # 计算线段方向向量
            seg_x = target_points_x[i] - missile_x
            seg_y = target_points_y[i] - missile_y
            seg_z = target_points_z[i] - missile_z
            
            segment_length_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
            
            if segment_length_sq == 0.0:
                # 如果目标点与导弹重合，直接计算距离
                dist = ((smoke_x - missile_x)**2 + (smoke_y - missile_y)**2 + (smoke_z - missile_z)**2)**0.5
            else:
                # 计算垂足
                to_x = smoke_x - missile_x
                to_y = smoke_y - missile_y
                to_z = smoke_z - missile_z
                
                t = (to_x * seg_x + to_y * seg_y + to_z * seg_z) / segment_length_sq
                t = max(0.0, min(1.0, t))
                
                foot_x = missile_x + t * seg_x
                foot_y = missile_y + t * seg_y
                foot_z = missile_z + t * seg_z
                
                # 计算距离
                dist = ((smoke_x - foot_x)**2 + (smoke_y - foot_y)**2 + (smoke_z - foot_z)**2)**0.5
            
            if dist > max_distance:
                max_distance = dist
        
        return max_distance
else:
    def _compute_max_distance_to_lines_numba(smoke_x, smoke_y, smoke_z, missile_x, missile_y, missile_z,
                                           target_points_x, target_points_y, target_points_z):
        max_distance = 0.0
        n_points = len(target_points_x)
        
        for i in range(n_points):
            seg_x, seg_y, seg_z = target_points_x[i] - missile_x, target_points_y[i] - missile_y, target_points_z[i] - missile_z
            segment_length_sq = seg_x * seg_x + seg_y * seg_y + seg_z * seg_z
            
            if segment_length_sq == 0.0:
                dist = ((smoke_x - missile_x)**2 + (smoke_y - missile_y)**2 + (smoke_z - missile_z)**2)**0.5
            else:
                to_x, to_y, to_z = smoke_x - missile_x, smoke_y - missile_y, smoke_z - missile_z
                t = max(0.0, min(1.0, (to_x * seg_x + to_y * seg_y + to_z * seg_z) / segment_length_sq))
                foot_x, foot_y, foot_z = missile_x + t * seg_x, missile_y + t * seg_y, missile_z + t * seg_z
                dist = ((smoke_x - foot_x)**2 + (smoke_y - foot_y)**2 + (smoke_z - foot_z)**2)**0.5
            
            max_distance = max(max_distance, dist)
        
        return max_distance


class MaskingCalculator:
    """遮蔽计算器"""
    
    def __init__(self):
        self.traj_calc = TrajectoryCalculator()
        self.threshold = SMOKE_PARAMS["cloud_radius"]
        self.time_step = CALCULATION_PARAMS["time_step"]
        self.max_time = CALCULATION_PARAMS["max_simulation_time"]
        
        # 预计算目标配置
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
        创建遮蔽判定函数 - 优化版本
        
        Args:
            missile_traj: 导弹轨迹函数
            smoke_traj: 烟幕云轨迹函数
            
        Returns:
            判定函数 predicate(t, threshold) -> bool
        """
        # 预计算目标中心
        base_center = np.array(self.target_centers[0], dtype=np.float64)
        top_center = np.array(self.target_centers[1], dtype=np.float64)
        
        @lru_cache(maxsize=1000)  # 缓存计算结果
        def cached_predicate(t: float, threshold: float) -> bool:
            # 获取当前时刻的导弹和烟幕云位置
            missile_pos = missile_traj(t)
            smoke_pos = smoke_traj(t)
            
            # 计算所有目标点（基于cyx实现：上下两层各3个点）
            top_points = get_top_plane_points(missile_pos, self.target_centers[1], self.target_radius)  # 上层
            under_points = get_under_points(missile_pos, self.target_centers[0], self.target_radius)   # 下层
            all_target_points = top_points + under_points
            
            # 转换为NumPy数组以便Numba优化
            target_points_x = np.array([p[0] for p in all_target_points], dtype=np.float64)
            target_points_y = np.array([p[1] for p in all_target_points], dtype=np.float64)
            target_points_z = np.array([p[2] for p in all_target_points], dtype=np.float64)
            
            # 使用优化版本计算最大距离
            max_distance = _compute_max_distance_to_lines_numba(
                smoke_pos[0], smoke_pos[1], smoke_pos[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                target_points_x, target_points_y, target_points_z
            )
            
            return max_distance <= threshold
        
        return cached_predicate
    
    def calculate_masking_duration(
        self,
        missile_traj: Callable[[float], Vector3],
        smoke_traj: Callable[[float], Vector3],
        start_time: float = 0.0,
        end_time: float = None,
        algorithm: str = "adaptive"  # "fixed", "adaptive", "smart"
    ) -> float:
        """
        计算有效遮蔽时长
        
        Args:
            missile_traj: 导弹轨迹函数
            smoke_traj: 烟幕云轨迹函数
            start_time: 开始时间
            end_time: 结束时间（如果为None则自动计算）
            algorithm: 区间查找算法 ("fixed", "adaptive", "smart")
            
        Returns:
            有效遮蔽时长（秒）
        """
        if end_time is None:
            # 自动计算合理的结束时间
            _, end_time = self.traj_calc.get_trajectory_bounds(missile_traj, self.max_time)
        
        # 创建遮蔽判定函数
        predicate = self._create_masking_predicate(missile_traj, smoke_traj)
        
        # 根据算法选择寻找满足遮蔽条件的时间区间
        if algorithm == "fixed":
            intervals = find_t_intervals(
                predicate, 
                self.threshold, 
                start_time, 
                end_time, 
                self.time_step
            )
        elif algorithm == "adaptive":
            intervals = find_t_intervals_adaptive(
                predicate,
                self.threshold,
                start_time,
                end_time,
                initial_step=self.time_step * 10,  # 开始时使用较大步长
                min_step=self.time_step / 2,      # 最小步长为原来的一半
                max_step=self.time_step * 50      # 最大步长
            )
        elif algorithm == "smart":
            intervals = find_t_intervals_smart(
                predicate,
                self.threshold,
                start_time,
                end_time,
                initial_step=self.time_step * 5,  # 智能算法的初始步长
                aggressive_speedup=True
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # 计算总遮蔽时长
        total_duration = sum(b - a for a, b in intervals)
        return total_duration


@lru_cache(maxsize=500)
def calculate_single_uav_single_smoke_masking(
    uav_direction: float,
    uav_speed: float,
    smoke_deploy_time: float,
    smoke_explode_delay: float,
    algorithm: str = "adaptive"  # "fixed", "adaptive", "smart"
) -> float:
    """
    计算单无人机单烟幕弹单导弹的有效遮蔽时长
    
    Args:
        uav_direction: 无人机飞行方向（度，0-360）
        uav_speed: 无人机飞行速度（m/s）
        smoke_deploy_time: 烟幕弹投放时间（s）
        smoke_explode_delay: 烟幕弹起爆相对延时（s）
        algorithm: 区间查找算法 ("fixed", "adaptive", "smart")
        
    Returns:
        有效遮蔽时长（s）
    """
    calc = MaskingCalculator()
    
    # 创建导弹轨迹
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 创建无人机轨迹
    uav_traj = calc.traj_calc.create_uav_trajectory("FY3", direction_degrees=uav_direction, speed=uav_speed)
    
    # 创建烟幕云轨迹
    smoke_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_deploy_time, smoke_explode_delay)
    
    # 计算遮蔽时长，使用指定的算法
    explode_time = smoke_deploy_time + smoke_explode_delay
    duration = calc.calculate_masking_duration(
        missile_traj, smoke_traj, 
        start_time=explode_time,
        algorithm=algorithm
    )
    
    return duration


def calculate_single_uav_triple_smoke_masking(
    uav_direction: float,
    uav_speed: float,
    smoke_a_deploy_time: float,
    smoke_a_explode_delay: float,
    smoke_b_deploy_delay: float,
    smoke_b_explode_delay: float,
    smoke_c_deploy_delay: float,
    smoke_c_explode_delay: float,
    algorithm: str = "adaptive"  # "fixed", "adaptive", "smart"
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
        algorithm: 区间查找算法 ("fixed", "adaptive", "smart")
        
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
    
    # 创建组合遮蔽判定函数 - 优化版本
    @lru_cache(maxsize=1000)
    def combined_predicate(t: float, threshold: float) -> bool:
        """组合判定：任一烟幕云满足遮蔽条件即可"""
        missile_pos = missile_traj(t)
        
        # 计算所有目标点（基于cyx实现：上下两层各3个点）
        top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)  # 上层
        under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)   # 下层
        all_target_points = top_points + under_points
        
        # 转换为NumPy数组
        target_points_x = np.array([p[0] for p in all_target_points], dtype=np.float64)
        target_points_y = np.array([p[1] for p in all_target_points], dtype=np.float64)
        target_points_z = np.array([p[2] for p in all_target_points], dtype=np.float64)
        
        # 检查每个烟幕云的遮蔽效果
        for smoke_traj in [smoke_a_traj, smoke_b_traj, smoke_c_traj]:
            smoke_pos = smoke_traj(t)
            max_distance = _compute_max_distance_to_lines_numba(
                smoke_pos[0], smoke_pos[1], smoke_pos[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                target_points_x, target_points_y, target_points_z
            )
            if max_distance <= threshold:
                return True
        
        return False
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 寻找满足条件的时间区间，使用指定算法
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    
    if algorithm == "fixed":
        intervals = find_t_intervals(
            combined_predicate,
            calc.threshold,
            earliest_explode,
            end_time,
            calc.time_step
        )
    elif algorithm == "adaptive":
        intervals = find_t_intervals_adaptive(
            combined_predicate,
            calc.threshold,
            earliest_explode,
            end_time,
            initial_step=calc.time_step * 10,
            min_step=calc.time_step / 2,
            max_step=calc.time_step * 50
        )
    elif algorithm == "smart":
        intervals = find_t_intervals_smart(
            combined_predicate,
            calc.threshold,
            earliest_explode,
            end_time,
            initial_step=calc.time_step * 5,
            aggressive_speedup=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
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
    
    # 创建组合遮蔽判定函数 - 优化版本
    @lru_cache(maxsize=1000)
    def combined_predicate(t: float, threshold: float) -> bool:
        """组合判定：任一烟幕云满足遮蔽条件即可"""
        missile_pos = missile_traj(t)
        
        # 计算所有目标点（基于cyx实现：上下两层各3个点）
        top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)  # 上层
        under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)   # 下层
        all_target_points = top_points + under_points
        
        # 转换为NumPy数组
        target_points_x = np.array([p[0] for p in all_target_points], dtype=np.float64)
        target_points_y = np.array([p[1] for p in all_target_points], dtype=np.float64)
        target_points_z = np.array([p[2] for p in all_target_points], dtype=np.float64)
        
        # 检查每个烟幕云的遮蔽效果
        for smoke_traj in [smoke_a_traj, smoke_b_traj, smoke_c_traj]:
            smoke_pos = smoke_traj(t)
            max_distance = _compute_max_distance_to_lines_numba(
                smoke_pos[0], smoke_pos[1], smoke_pos[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                target_points_x, target_points_y, target_points_z
            )
            if max_distance <= threshold:
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
    # 导入联合遮挡计算函数
    from .for_merge import find_full_cover_intervals
    
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
    
    # 创建多烟幕弹轨迹函数（返回所有活跃烟幕弹位置）
    def multi_smoke_trajectory(t: float) -> List[Vector3]:
        """返回当前时刻所有活跃烟幕弹的位置列表"""
        active_smoke_positions = []
        
        # 检查烟幕弹A是否活跃
        if t >= smoke_a_deploy_time + smoke_a_explode_delay:
            active_smoke_positions.append(smoke_a_traj(t))
        
        # 检查烟幕弹B是否活跃
        if t >= smoke_b_deploy_time + smoke_b_explode_delay:
            active_smoke_positions.append(smoke_b_traj(t))
        
        # 检查烟幕弹C是否活跃
        if t >= smoke_c_deploy_time + smoke_c_explode_delay:
            active_smoke_positions.append(smoke_c_traj(t))
        
        return active_smoke_positions
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 获取时间边界
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    
    # 使用 find_full_cover_intervals 计算联合遮挡时间区间
    intervals = find_full_cover_intervals(
        traj0_fn=missile_traj,                    # 导弹轨迹
        traj1_fn=multi_smoke_trajectory,          # 多烟幕弹轨迹
        t_min=earliest_explode,                   # 开始时间
        t_max=end_time,                           # 结束时间
        step=calc.time_step,                      # 时间步长
        sphere_radius_for_cone=calc.threshold     # 烟幕云半径
    )
    
    # 计算总遮蔽时长
    total_duration = sum(b - a for a, b in intervals)
    return total_duration


def calculate_multi_uav_single_smoke_masking_multiple(
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
    计算多无人机多烟幕弹单导弹的有效遮蔽时长 - 联合遮挡版本
    
    与 calculate_multi_uav_single_smoke_masking 的区别：
    - 原版本：任意一个烟幕弹独立满足遮挡条件即可
    - 此版本：考虑多个烟幕弹联合遮挡的协同效应
    
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
    # 导入联合遮挡计算函数
    from .for_merge import find_full_cover_intervals
    
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
    
    # 创建多烟幕弹轨迹函数（返回所有活跃烟幕弹位置）
    def multi_smoke_trajectory(t: float) -> List[Vector3]:
        """返回当前时刻所有活跃烟幕弹的位置列表"""
        active_smoke_positions = []
        
        # 检查烟幕弹A是否活跃
        if t >= smoke_a_deploy_time + smoke_a_explode_delay:
            active_smoke_positions.append(smoke_a_traj(t))
        
        # 检查烟幕弹B是否活跃
        if t >= smoke_b_deploy_time + smoke_b_explode_delay:
            active_smoke_positions.append(smoke_b_traj(t))
        
        # 检查烟幕弹C是否活跃
        if t >= smoke_c_deploy_time + smoke_c_explode_delay:
            active_smoke_positions.append(smoke_c_traj(t))
        
        return active_smoke_positions
    
    # 计算最早起爆时间作为开始时间
    earliest_explode = min(
        smoke_a_deploy_time + smoke_a_explode_delay,
        smoke_b_deploy_time + smoke_b_explode_delay,
        smoke_c_deploy_time + smoke_c_explode_delay
    )
    
    # 获取时间边界
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    
    # 使用 find_full_cover_intervals 计算联合遮挡时间区间
    intervals = find_full_cover_intervals(
        traj0_fn=missile_traj,                    # 导弹轨迹
        traj1_fn=multi_smoke_trajectory,          # 多烟幕弹轨迹
        t_min=earliest_explode,                   # 开始时间
        t_max=end_time,                           # 结束时间
        step=calc.time_step,                      # 时间步长
        sphere_radius_for_cone=calc.threshold     # 烟幕云半径
    )
    
    # 计算总遮蔽时长
    total_duration = sum(b - a for a, b in intervals)
    return total_duration


# 所有计算函数已在上面定义完成

def calculate_problem5_multi_uav_multi_missile_masking(
    # 5架无人机参数
    uav_1_direction: float, uav_1_speed: float,
    uav_2_direction: float, uav_2_speed: float,
    uav_3_direction: float, uav_3_speed: float,
    uav_4_direction: float, uav_4_speed: float,
    uav_5_direction: float, uav_5_speed: float,
    # 每架无人机3枚烟幕弹参数 (15枚烟幕弹)
    uav_1_smoke_1_deploy_time: float, uav_1_smoke_1_explode_delay: float,
    uav_1_smoke_2_deploy_time: float, uav_1_smoke_2_explode_delay: float,
    uav_1_smoke_3_deploy_time: float, uav_1_smoke_3_explode_delay: float,
    uav_2_smoke_1_deploy_time: float, uav_2_smoke_1_explode_delay: float,
    uav_2_smoke_2_deploy_time: float, uav_2_smoke_2_explode_delay: float,
    uav_2_smoke_3_deploy_time: float, uav_2_smoke_3_explode_delay: float,
    uav_3_smoke_1_deploy_time: float, uav_3_smoke_1_explode_delay: float,
    uav_3_smoke_2_deploy_time: float, uav_3_smoke_2_explode_delay: float,
    uav_3_smoke_3_deploy_time: float, uav_3_smoke_3_explode_delay: float,
    uav_4_smoke_1_deploy_time: float, uav_4_smoke_1_explode_delay: float,
    uav_4_smoke_2_deploy_time: float, uav_4_smoke_2_explode_delay: float,
    uav_4_smoke_3_deploy_time: float, uav_4_smoke_3_explode_delay: float,
    uav_5_smoke_1_deploy_time: float, uav_5_smoke_1_explode_delay: float,
    uav_5_smoke_2_deploy_time: float, uav_5_smoke_2_explode_delay: float,
    uav_5_smoke_3_deploy_time: float, uav_5_smoke_3_explode_delay: float
) -> float:
    """
    计算第五问：5架无人机每架最多3枚烟幕弹对3枚导弹的综合遮蔽效果
    
    目标：最大化对M1、M2、M3三枚导弹的综合遮蔽时长
    
    Args:
        uav_*_direction: 无人机飞行方向（度，0-360）
        uav_*_speed: 无人机飞行速度（m/s）
        uav_*_smoke_*_deploy_time: 烟幕弹投放时间（s）
        uav_*_smoke_*_explode_delay: 烟幕弹起爆延时（s）
        
    Returns:
        综合遮蔽效果评分（考虑所有导弹的遮蔽时长）
    """
    calc = MaskingCalculator()
    
    # 创建三枚导弹的轨迹
    missile_trajs = {
        'M1': calc.traj_calc.create_missile_trajectory("M1"),
        'M2': calc.traj_calc.create_missile_trajectory("M2"),
        'M3': calc.traj_calc.create_missile_trajectory("M3")
    }
    
    # 创建5架无人机轨迹
    uav_params = [
        (uav_1_direction, uav_1_speed, "FY1"),
        (uav_2_direction, uav_2_speed, "FY2"),
        (uav_3_direction, uav_3_speed, "FY3"),
        (uav_4_direction, uav_4_speed, "FY4"),
        (uav_5_direction, uav_5_speed, "FY5")
    ]
    
    uav_trajs = {}
    for i, (direction, speed, uav_id) in enumerate(uav_params):
        uav_trajs[f'UAV_{i+1}'] = calc.traj_calc.create_uav_trajectory(
            uav_id, direction_degrees=direction, speed=speed
        )
    
    # 创建所有烟幕弹轨迹（15枚烟幕弹）
    smoke_params = [
        # UAV1的3枚烟幕弹
        (uav_1_smoke_1_deploy_time, uav_1_smoke_1_explode_delay, 'UAV_1'),
        (uav_1_smoke_2_deploy_time, uav_1_smoke_2_explode_delay, 'UAV_1'),
        (uav_1_smoke_3_deploy_time, uav_1_smoke_3_explode_delay, 'UAV_1'),
        # UAV2的3枚烟幕弹
        (uav_2_smoke_1_deploy_time, uav_2_smoke_1_explode_delay, 'UAV_2'),
        (uav_2_smoke_2_deploy_time, uav_2_smoke_2_explode_delay, 'UAV_2'),
        (uav_2_smoke_3_deploy_time, uav_2_smoke_3_explode_delay, 'UAV_2'),
        # UAV3的3枚烟幕弹
        (uav_3_smoke_1_deploy_time, uav_3_smoke_1_explode_delay, 'UAV_3'),
        (uav_3_smoke_2_deploy_time, uav_3_smoke_2_explode_delay, 'UAV_3'),
        (uav_3_smoke_3_deploy_time, uav_3_smoke_3_explode_delay, 'UAV_3'),
        # UAV4的3枚烟幕弹
        (uav_4_smoke_1_deploy_time, uav_4_smoke_1_explode_delay, 'UAV_4'),
        (uav_4_smoke_2_deploy_time, uav_4_smoke_2_explode_delay, 'UAV_4'),
        (uav_4_smoke_3_deploy_time, uav_4_smoke_3_explode_delay, 'UAV_4'),
        # UAV5的3枚烟幕弹
        (uav_5_smoke_1_deploy_time, uav_5_smoke_1_explode_delay, 'UAV_5'),
        (uav_5_smoke_2_deploy_time, uav_5_smoke_2_explode_delay, 'UAV_5'),
        (uav_5_smoke_3_deploy_time, uav_5_smoke_3_explode_delay, 'UAV_5')
    ]
    
    smoke_trajs = []
    for deploy_time, explode_delay, uav_key in smoke_params:
        smoke_traj = calc.traj_calc.create_smoke_trajectory(
            uav_trajs[uav_key], deploy_time, explode_delay
        )
        smoke_trajs.append((smoke_traj, deploy_time + explode_delay))
    
    # 计算每枚导弹的遮蔽效果
    total_masking_score = 0.0
    missile_weights = {'M1': 1.0, 'M2': 1.0, 'M3': 1.0}  # 可以调整权重
    
    for missile_id, missile_traj in missile_trajs.items():
        # 为当前导弹创建组合遮蔽判定函数
        @lru_cache(maxsize=1000)
        def missile_predicate(t: float, threshold: float) -> bool:
            """组合判定：任一烟幕云满足遮蔽条件即可"""
            missile_pos = missile_traj(t)
            
            # 计算所有目标点
            top_points = get_top_plane_points(missile_pos, calc.target_centers[1], calc.target_radius)
            under_points = get_under_points(missile_pos, calc.target_centers[0], calc.target_radius)
            all_target_points = top_points + under_points
            
            # 转换为NumPy数组
            target_points_x = np.array([p[0] for p in all_target_points], dtype=np.float64)
            target_points_y = np.array([p[1] for p in all_target_points], dtype=np.float64)
            target_points_z = np.array([p[2] for p in all_target_points], dtype=np.float64)
            
            # 检查每个活跃烟幕云的遮蔽效果
            for smoke_traj, explode_time in smoke_trajs:
                if t >= explode_time:  # 烟幕弹已起爆
                    smoke_pos = smoke_traj(t)
                    max_distance = _compute_max_distance_to_lines_numba(
                        smoke_pos[0], smoke_pos[1], smoke_pos[2],
                        missile_pos[0], missile_pos[1], missile_pos[2],
                        target_points_x, target_points_y, target_points_z
                    )
                    if max_distance <= threshold:
                        return True
            
            return False
        
        # 计算最早起爆时间
        earliest_explode = min(explode_time for _, explode_time in smoke_trajs)
        
        # 计算时间边界
        _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
        
        # 寻找满足条件的时间区间
        intervals = find_t_intervals_adaptive(
            missile_predicate,
            calc.threshold,
            earliest_explode,
            end_time,
            initial_step=calc.time_step * 10,
            min_step=calc.time_step / 2,
            max_step=calc.time_step * 50
        )
        
        # 计算当前导弹的遮蔽时长
        missile_masking_duration = sum(b - a for a, b in intervals)
        total_masking_score += missile_weights[missile_id] * missile_masking_duration
        
        # 清空缓存以避免内存泄漏
        missile_predicate.cache_clear()
    
    return total_masking_score
