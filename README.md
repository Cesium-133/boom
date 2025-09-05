# CUMCM2025 Problem A Solver - 导弹拦截烟幕遮蔽分析

## 项目概述

本项目是针对2025年全国大学生数学建模竞赛A题的解决方案，主要功能是分析导弹拦截过程中烟幕弹的有效遮蔽时长计算。

## 项目结构

```
CUMCM2025-Problem-A-Solver/
├── A题.pdf                    # 竞赛题目文档
├── README.md                  # 项目说明文档
├── solver/                    # 核心求解模块
│   ├── masking/              # 遮蔽计算模块
│   │   ├── __init__.py
│   │   ├── config.py         # 配置参数
│   │   ├── core.py           # 核心计算逻辑
│   │   ├── trajectory.py     # 轨迹计算
│   │   └── geometry.py       # 几何计算工具
├── cyx/                      # 参考实现代码
├── missile_simulation/       # 可视化仿真模块
└── 附件/                     # 数据文件
```

## 核心功能

### 1. 有效遮蔽时长计算

本项目实现了三种主要的遮蔽计算场景：

#### 1.1 单无人机单烟幕弹单导弹

- **函数**: `calculate_single_uav_single_smoke_masking`
- **输入参数**:
  - `uav_direction`: 无人机飞行方向（度，0-360）
  - `uav_speed`: 无人机飞行速度（m/s）
  - `smoke_deploy_time`: 烟幕弹投放时间（s）
  - `smoke_explode_delay`: 烟幕弹起爆相对延时（s）
- **输出**: 有效遮蔽时长（s）

#### 1.2 单无人机3烟幕弹单导弹

- **函数**: `calculate_single_uav_triple_smoke_masking`
- **输入参数**:
  - `uav_direction`: 无人机飞行方向（度）
  - `uav_speed`: 无人机飞行速度（m/s）
  - `smoke_a_deploy_time`: 烟幕弹A投放时间（s）
  - `smoke_a_explode_delay`: 烟幕弹A起爆延时（s）
  - `smoke_b_deploy_delay`: 烟幕弹B相对A的投放延时（s）
  - `smoke_b_explode_delay`: 烟幕弹B起爆延时（s）
  - `smoke_c_deploy_delay`: 烟幕弹C相对B的投放延时（s）
  - `smoke_c_explode_delay`: 烟幕弹C起爆延时（s）
- **输出**: 有效遮蔽时长（s）

#### 1.3 多无人机单烟幕弹单导弹

- **函数**: `calculate_multi_uav_single_smoke_masking`
- **输入参数**:
  - `uav_a_direction`, `uav_a_speed`: 无人机A参数
  - `uav_b_direction`, `uav_b_speed`: 无人机B参数
  - `uav_c_direction`, `uav_c_speed`: 无人机C参数
  - `smoke_a_deploy_time`, `smoke_a_explode_delay`: 烟幕弹A参数
  - `smoke_b_deploy_time`, `smoke_b_explode_delay`: 烟幕弹B参数
  - `smoke_c_deploy_time`, `smoke_c_explode_delay`: 烟幕弹C参数
- **输出**: 有效遮蔽时长（s）

## 使用方法

### 基本使用

```python
from solver.masking import calculate_single_uav_single_smoke_masking

# 计算单无人机单烟幕弹的有效遮蔽时长
masking_duration = calculate_single_uav_single_smoke_masking(
    uav_direction=90,        # 无人机向北飞行
    uav_speed=120,           # 速度120m/s
    smoke_deploy_time=5.0,   # 5秒时投放烟幕弹
    smoke_explode_delay=1.5  # 投放后1.5秒起爆
)

print(f"有效遮蔽时长: {masking_duration:.2f}秒")
```

### 配置参数

系统默认配置参数包括：

- 重力加速度: 9.8 m/s²
- 导弹速度: 300 m/s
- 烟幕云半径: 10 m
- 烟幕云下沉速度: 3 m/s

可通过修改 `solver/masking/config.py` 调整参数。

## 技术原理

### 遮蔽判定原理

1. **轨迹计算**: 基于物理模型计算导弹和无人机的运动轨迹
2. **几何分析**: 计算烟幕云与导弹-目标视线的几何关系
3. **时间区间**: 使用二分法精确计算满足遮蔽条件的时间区间
4. **阈值判定**: 当所有关键点到视线的距离小于烟幕半径时认为有效遮蔽

### 核心算法

- 采用扫描+二分法寻找满足遮蔽条件的时间区间
- 精确计算点到直线的最短距离
- 动态更新目标点位置以适应导弹运动轨迹

## 注意事项

1. 所有角度输入使用度数制（0-360度）
2. 坐标系采用右手坐标系，z轴向上
3. 时间参数均为相对时间，单位为秒
4. 遮蔽效果基于几何遮挡模型，未考虑风力等环境因素
