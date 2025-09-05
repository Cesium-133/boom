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
    "time_step": 0.01,                  # 时间步长 (s)
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
