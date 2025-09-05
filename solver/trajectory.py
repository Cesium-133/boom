"""
轨迹计算模块

提供导弹、无人机和烟幕弹的轨迹计算功能。
基于物理模型计算各种载具的运动轨迹。
"""

from __future__ import annotations
from typing import Tuple, Callable
from math import sqrt
from .config import CONSTANTS, MISSILES, UAVS, TARGETS, SMOKE_PARAMS
from .geometry import Vector3, distance_to_origin, direction_to_unit_vector


class TrajectoryCalculator:
    """轨迹计算器"""
    
    def __init__(self):
        self.g = CONSTANTS["g"]
        self.smoke_sink_speed = SMOKE_PARAMS["cloud_sink_speed"]
    
    def create_missile_trajectory(self, missile_id: str = "M1") -> Callable[[float], Vector3]:
        """
        创建导弹轨迹函数 - 简化版本
        
        Args:
            missile_id: 导弹标识符
            
        Returns:
            轨迹函数 traj(t) -> (x, y, z)
        """
        missile_config = MISSILES[missile_id]
        p0 = missile_config["initial_pos"]  # 导弹初始位置
        missile_speed = missile_config["speed"]  # 导弹速度
        
        # 目标是虚假目标（原点）
        target = TARGETS["fake_target"]  # [0, 0, 0]
        
        # 计算从导弹初始位置指向目标的单位方向向量
        direction_x = target[0] - p0[0]  # 0 - 20000 = -20000
        direction_y = target[1] - p0[1]  # 0 - 0 = 0  
        direction_z = target[2] - p0[2]  # 0 - 2000 = -2000
        
        # 单位化方向向量
        distance = sqrt(direction_x**2 + direction_y**2 + direction_z**2)
        ux = direction_x / distance
        uy = direction_y / distance
        uz = direction_z / distance
        
        def missile_trajectory(t: float) -> Vector3:
            """导弹轨迹：从初始位置匀速飞向虚假目标（原点）"""
            return (
                p0[0] + missile_speed * t * ux,
                p0[1] + missile_speed * t * uy,
                p0[2] + missile_speed * t * uz,
            )
        
        return missile_trajectory
    
    def create_uav_trajectory(
        self, 
        uav_id: str = "FY1",
        direction_degrees: float = None,
        speed: float = 120
    ) -> Callable[[float], Vector3]:
        """
        创建无人机轨迹函数 - 朝向虚假目标水平飞行
        
        Args:
            uav_id: 无人机标识符
            direction_degrees: 飞行方向（度，如果为None则自动计算朝向虚假目标的方向）
            speed: 飞行速度（m/s）
            
        Returns:
            轨迹函数 traj(t) -> (x, y, z)
        """
        uav_config = UAVS[uav_id]
        p1 = uav_config["initial_pos"]  # 无人机初始位置
        fake_target = TARGETS["fake_target"]  # 虚假目标位置 [0, 0, 0]
        
        if direction_degrees is None:
            # 自动计算朝向虚假目标的方向（水平方向）
            dx = fake_target[0] - p1[0]
            dy = fake_target[1] - p1[1]
            distance_xy = sqrt(dx**2 + dy**2)
            if distance_xy > 0:
                ux = dx / distance_xy
                uy = dy / distance_xy
            else:
                ux, uy = 1.0, 0.0
        else:
            # 使用指定的方向
            ux, uy = direction_to_unit_vector(direction_degrees)
        
        def uav_trajectory(t: float) -> Vector3:
            """无人机轨迹：朝向虚假目标的水平匀速直线运动"""
            return (
                p1[0] + speed * t * ux,  # x方向：初始位置 + 速度*时间*方向
                p1[1] + speed * t * uy,  # y方向：初始位置 + 速度*时间*方向
                p1[2],  # z方向：保持水平飞行高度不变
            )
        
        return uav_trajectory
    
    def create_smoke_trajectory(
        self,
        uav_trajectory: Callable[[float], Vector3],
        deploy_time: float,
        explode_delay: float
    ) -> Callable[[float], Vector3]:
        """
        创建烟幕弹轨迹函数
        
        Args:
            uav_trajectory: 无人机轨迹函数
            deploy_time: 烟幕弹投放时间
            explode_delay: 起爆延时
            
        Returns:
            烟幕云中心轨迹函数 traj(t) -> (x, y, z)
        """
        # 计算投放位置和速度
        deploy_pos = uav_trajectory(deploy_time)
        dt = 0.001  # 小时间增量用于计算速度
        deploy_pos_next = uav_trajectory(deploy_time + dt)
        
        # 计算无人机水平速度
        vx = (deploy_pos_next[0] - deploy_pos[0]) / dt
        vy = (deploy_pos_next[1] - deploy_pos[1]) / dt
        
        explode_time = deploy_time + explode_delay
        
        def smoke_trajectory(t: float) -> Vector3:
            """烟幕弹轨迹：投放后平抛运动，起爆后烟幕云竖直下沉"""
            if t < deploy_time:
                # 投放前，在无人机上
                return uav_trajectory(t)
            elif t < explode_time:
                # 投放后到起爆前：平抛运动
                flight_time = t - deploy_time
                return (
                    deploy_pos[0] + vx * flight_time,  # 水平匀速
                    deploy_pos[1] + vy * flight_time,  # 水平匀速
                    deploy_pos[2] - 0.5 * self.g * flight_time**2  # 自由落体
                )
            else:
                # 起爆后：烟幕云从起爆位置竖直下沉
                # 先计算起爆位置
                flight_time = explode_delay
                explode_x = deploy_pos[0] + vx * flight_time
                explode_y = deploy_pos[1] + vy * flight_time
                explode_z = deploy_pos[2] - 0.5 * self.g * flight_time**2
                
                # 烟幕云下沉
                sink_time = t - explode_time
                return (
                    explode_x,  # x位置固定
                    explode_y,  # y位置固定
                    explode_z - self.smoke_sink_speed * sink_time  # 竖直下沉
                )
        
        return smoke_trajectory
    
    def get_trajectory_bounds(self, trajectory: Callable[[float], Vector3], max_time: float = 100) -> Tuple[float, float]:
        """
        获取轨迹的合理时间边界
        
        Args:
            trajectory: 轨迹函数
            max_time: 最大时间限制
            
        Returns:
            (t_min, t_max): 时间范围
        """
        t_min = 0.0
        t_max = max_time
        
        # 检查轨迹是否到达地面或原点附近
        for t in [i * 0.1 for i in range(int(max_time * 10))]:
            pos = trajectory(t)
            if pos[2] <= 0:  # 到达地面
                t_max = min(t_max, t)
                break
            if distance_to_origin(pos) < 100:  # 接近原点
                t_max = min(t_max, t + 10)  # 给一些余量
                break
                
        return t_min, t_max
