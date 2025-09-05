"""
烟幕遮蔽计算核心模块 - Numba 优化版本

主要优化点：
1. 遮蔽判定函数的核心循环
2. 距离计算的批量处理
3. 减少Python对象创建和函数调用开销
"""

import numpy as np
from numba import jit
from typing import List, Tuple, Callable
from functools import lru_cache
from .config import CONSTANTS, TARGETS, CALCULATION_PARAMS, SMOKE_PARAMS
from .geometry_optimized import (
    compute_masking_distances_numba, 
    get_circle_points_numba,
    distance_between_numba
)
from .trajectory import TrajectoryCalculator

class OptimizedMaskingCalculator:
    """优化版遮蔽计算器"""
    
    def __init__(self):
        self.traj_calc = TrajectoryCalculator()
        self.threshold = SMOKE_PARAMS["cloud_radius"]
        self.time_step = CALCULATION_PARAMS["time_step"]
        self.max_time = CALCULATION_PARAMS["max_simulation_time"]
        
        # 预计算目标配置
        self.base_center = np.array(TARGETS["true_target"]["base_center"], dtype=np.float64)
        self.top_center = self.base_center.copy()
        self.top_center[2] += TARGETS["true_target"]["height"]
        self.target_radius = TARGETS["true_target"]["radius"]
    
    def _create_optimized_masking_predicate(self, 
                                          missile_traj: Callable[[float], Tuple[float, float, float]],
                                          smoke_traj: Callable[[float], Tuple[float, float, float]]):
        """创建优化版遮蔽判定函数"""
        
        @lru_cache(maxsize=1000)  # 缓存计算结果
        def cached_predicate(t: float, threshold: float) -> bool:
            # 获取当前位置
            missile_pos = missile_traj(t)
            smoke_pos = smoke_traj(t)
            
            # 获取所有目标点（使用优化版本）
            top_points_x, top_points_y, top_points_z = get_circle_points_numba(
                self.top_center[0], self.top_center[1], self.top_center[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                self.target_radius
            )
            
            under_points_x, under_points_y, under_points_z = get_circle_points_numba(
                self.base_center[0], self.base_center[1], self.base_center[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                self.target_radius
            )
            
            # 合并所有目标点
            all_points_x = np.concatenate([top_points_x, under_points_x])
            all_points_y = np.concatenate([top_points_y, under_points_y])
            all_points_z = np.concatenate([top_points_z, under_points_z])
            
            # 使用优化版本计算最大距离
            max_distance = compute_masking_distances_numba(
                smoke_pos[0], smoke_pos[1], smoke_pos[2],
                missile_pos[0], missile_pos[1], missile_pos[2],
                all_points_x, all_points_y, all_points_z
            )
            
            return max_distance <= threshold
        
        return cached_predicate
    
    def calculate_masking_duration(self,
                                 missile_traj: Callable[[float], Tuple[float, float, float]],
                                 smoke_traj: Callable[[float], Tuple[float, float, float]],
                                 start_time: float = 0.0,
                                 end_time: float = None) -> float:
        """计算有效遮蔽时长 - 优化版本"""
        
        if end_time is None:
            end_time = self.max_time
        
        # 创建优化版遮蔽判定函数
        predicate = self._create_optimized_masking_predicate(missile_traj, smoke_traj)
        
        # 使用优化版区间查找
        from .geometry import find_t_intervals
        intervals = find_t_intervals(
            predicate, self.threshold, start_time, end_time, self.time_step
        )
        
        # 计算总时长
        total_duration = sum(end - start for start, end in intervals)
        return total_duration

# 优化版主函数
@lru_cache(maxsize=500)  # 缓存函数结果
def calculate_single_uav_single_smoke_masking_optimized(
    uav_direction: float,
    uav_speed: float,
    smoke_deploy_time: float,
    smoke_explode_delay: float
) -> float:
    """
    优化版单无人机单烟幕弹遮蔽时长计算
    
    性能优化：
    1. 使用 @lru_cache 缓存相同参数的计算结果
    2. 使用优化版遮蔽计算器
    3. 减少不必要的对象创建
    """
    calculator = OptimizedMaskingCalculator()
    
    # 创建轨迹函数
    missile_traj = calculator.traj_calc.create_missile_trajectory("M1")
    uav_traj = calculator.traj_calc.create_uav_trajectory(
        "FY1", direction_degrees=uav_direction, speed=uav_speed
    )
    smoke_traj = calculator.traj_calc.create_smoke_trajectory(
        uav_traj, smoke_deploy_time, smoke_explode_delay
    )
    
    # 计算遮蔽时长
    explode_time = smoke_deploy_time + smoke_explode_delay
    end_time = explode_time + SMOKE_PARAMS["duration"]
    
    return calculator.calculate_masking_duration(
        missile_traj, smoke_traj, explode_time, end_time
    )