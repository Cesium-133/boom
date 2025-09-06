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

### 可视化调参 UI（Streamlit）

- 启动命令：

```bash
python -m streamlit run scripts/tuner_ui.py
```

- 功能：
  - 实时曲线：最佳适应度、种群多样性
  - 散点分布：`t_deploy` 与 `t_fuse` 的当前种群分布（颜色代表适应度）
  - 局部最优性：围绕当前最优解的二维邻域热力图，检测是否“近似局部最优”

- Windows 提示：为避免多进程在动态模块加载上的问题，UI 默认关闭并行开关；如需开启请在侧栏勾选。

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

### 高级优化器：iL-SHADE

- **脚本**: `Problem2-iL-SHADE.py`
- **算法**: iL-SHADE（L-SHADE 改进版，current-to-pbest/1+Archive，自适应F/CR，线性种群收缩）
- **用法**:
```bash
python Problem2-iL-SHADE.py
```
- **关键参数**:
  - `population_size_init`: 初始种群规模（默认80）
  - `population_size_min`: 最小种群规模（默认20）
  - `max_generations`: 最大迭代代数（默认800）
  - `memory_size`: 参数记忆槽规模H（默认20）
  - `p_best_max`: pbest 上界（默认0.2，pmin自适应为2/NP）
  - `use_parallel`: 是否并行评估（默认True）

- **输出**: 控制台打印最佳适应度与对应参数，返回优化过程记录

### 服务器版本：80核心并行

- **脚本**: `Problem4-DE-Server.py`
- **用途**: 针对80核心服务器优化的高性能版本
- **特性**:
  - 80个个体完全并行计算（每个体一个进程）
  - 自动检查点保存和恢复机制
  - 实时性能监控和资源使用统计
  - 优雅的中断处理和错误恢复
  - 支持长时间运行（2000代+）

- **服务器要求**:
  - CPU: 80+ 核心
  - RAM: 32GB+
  - Python 3.8+
  - 推荐操作系统: Linux

- **用法**:
```bash
python Problem4-DE-Server.py
```

- **性能目标**:
  - 每代评估时间: < 100ms
  - 每个体平均时间: < 1.5ms
  - 并行效率: > 70%
  - 理论加速比: 80x

- **检查点功能**:
  - 自动保存: 每50代保存一次检查点
  - 恢复运行: 支持从任意检查点恢复优化
  - 异常保护: 意外中断时自动保存当前最优解
