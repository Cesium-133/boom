"""
导弹拦截烟幕遮蔽分析系统配置文件

定义了系统中使用的所有常量、目标参数、导弹参数和无人机参数。
所有参数基于题目要求和物理模型设定。
"""

from typing import Dict, List, Tuple, Any

# 基础物理常量
CONSTANTS = {
    "g": 9.8,                          # 重力加速度 (m/s²)
}

# 目标参数配置
TARGETS = {
    "fake_target": [0, 0, 0],           # 虚假目标坐标 (x, y, z)
    "true_target": {
        "base_center": [0, 200, 0],     # 真实目标基础中心 (x, y, z)
        "radius": 7,                    # 目标半径 (m)
        "height": 10                    # 目标高度 (m)
    }
}

# 导弹配置（根据题目补全所有导弹）
MISSILES = {
    "M1": {
        "initial_pos": [20000, 0, 2000], # 导弹初始位置 (x, y, z)
        "speed": 300,                    # 导弹速度 (m/s)
    },
    "M2": {
        "initial_pos": [19000, 600, 2100], # 导弹2初始位置
        "speed": 300,                    # 导弹速度 (m/s)
    },
    "M3": {
        "initial_pos": [18000, -600, 1900], # 导弹3初始位置
        "speed": 300,                    # 导弹速度 (m/s)
    }
}

# 无人机配置（根据题目补全所有无人机）
UAVS = {
    "FY1": {
        "initial_pos": [17800, 0, 1800], # 无人机初始位置 (x, y, z)
        "speed_range": [70, 140],        # 速度范围 (m/s)
    },
    "FY2": {
        "initial_pos": [12000, 1400, 1400], # 无人机2初始位置
        "speed_range": [70, 140],        # 速度范围 (m/s)
    },
    "FY3": {
        "initial_pos": [6000, -3000, 700], # 无人机3初始位置
        "speed_range": [70, 140],        # 速度范围 (m/s)
    },
    "FY4": {
        "initial_pos": [11000, 2000, 1800], # 无人机4初始位置
        "speed_range": [70, 140],        # 速度范围 (m/s)
    },
    "FY5": {
        "initial_pos": [13000, -2000, 1300],  # 无人机5初始位置
        "speed_range": [70, 140],        # 速度范围 (m/s)
    }
}

# 计算参数
CALCULATION_PARAMS = {
    "time_step": 0.1,                  # 时间步长 (s)
    "max_simulation_time": 100.0,       # 最大仿真时间 (s)
    "masking_threshold": 10.0,          # 遮蔽阈值距离 (m)
    "boundary_refinement_iters": 25     # 边界细化迭代次数
}

# 烟幕弹参数
SMOKE_PARAMS = {
    "cloud_radius": 10,                 # 烟幕云半径 (m)
    "cloud_sink_speed": 3,              # 烟幕云下沉速度 (m/s)
    "duration": 20                      # 烟幕持续时间 (s)
}

# 第五问优化参数边界定义
def calculate_problem5_bounds():
    """计算第五问的搜索空间边界"""
    import numpy as np
    from .trajectory import TrajectoryCalculator
    
    traj_calc = TrajectoryCalculator()
    
    # 计算三枚导弹到达虚假目标的最大时间
    max_times = []
    for missile_id in ["M1", "M2", "M3"]:
        missile_pos = MISSILES[missile_id]["initial_pos"]
        missile_speed = MISSILES[missile_id]["speed"]
        fake_target = TARGETS["fake_target"]
        
        distance = np.sqrt(
            (fake_target[0] - missile_pos[0])**2 + 
            (fake_target[1] - missile_pos[1])**2 + 
            (fake_target[2] - missile_pos[2])**2
        )
        t_max = distance / missile_speed
        max_times.append(t_max)
    
    # 使用最大时间作为时间边界
    global_t_max = max(max_times)
    
    bounds = {}
    
    # 5架无人机参数（每架无人机最多投放3枚烟幕弹）
    for uav_idx in range(5):  # FY1-FY5
        uav_name = f"uav_{uav_idx+1}"
        bounds[f'{uav_name}_direction'] = (0.0, 360.0)      # 飞行方向
        bounds[f'{uav_name}_speed'] = (70.0, 140.0)         # 飞行速度
        
        # 每架无人机最多3枚烟幕弹
        for smoke_idx in range(3):
            smoke_name = f"{uav_name}_smoke_{smoke_idx+1}"
            bounds[f'{smoke_name}_deploy_time'] = (0.01, global_t_max - 2.0)  # 投放时间
            bounds[f'{smoke_name}_explode_delay'] = (0.01, 8.0)               # 起爆延时
    
    return bounds

# 为向后兼容，定义BOUNDS常量
BOUNDS = calculate_problem5_bounds()
