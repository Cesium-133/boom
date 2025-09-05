"""
轨迹可视化模块

用于验证和调试轨迹计算的正确性
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from .trajectory import TrajectoryCalculator
from .geometry import get_top_plane_points, get_under_points
from .config import TARGETS

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def visualize_trajectories(
    missile_id: str = "M1",
    uav_id: str = "FY1", 
    uav_direction: float = 90,
    uav_speed: float = 120,
    time_range: tuple = (0, 10),
    num_points: int = 100
):
    """
    可视化导弹和无人机轨迹
    
    Args:
        missile_id: 导弹标识符
        uav_id: 无人机标识符
        uav_direction: 无人机飞行方向（度）
        uav_speed: 无人机速度
        time_range: 时间范围 (开始时间, 结束时间)
        num_points: 绘制点数
    """
    traj_calc = TrajectoryCalculator()
    
    # 创建轨迹函数
    missile_traj = traj_calc.create_missile_trajectory(missile_id)
    uav_traj = traj_calc.create_uav_trajectory(uav_id, uav_direction, uav_speed)
    
    # 生成时间点
    t_start, t_end = time_range
    times = np.linspace(t_start, t_end, num_points)
    
    # 计算轨迹点
    missile_points = [missile_traj(t) for t in times]
    uav_points = [uav_traj(t) for t in times]
    
    # 提取坐标
    missile_x = [p[0] for p in missile_points]
    missile_y = [p[1] for p in missile_points]
    missile_z = [p[2] for p in missile_points]
    
    uav_x = [p[0] for p in uav_points]
    uav_y = [p[1] for p in uav_points]
    uav_z = [p[2] for p in uav_points]
    
    # 创建3D图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制轨迹
    ax.plot(missile_x, missile_y, missile_z, 'b-', linewidth=2, label=f'导弹{missile_id}轨迹')
    ax.plot(uav_x, uav_y, uav_z, 'r-', linewidth=2, label=f'无人机{uav_id}轨迹')
    
    # 标记起点和终点
    ax.scatter([missile_x[0]], [missile_y[0]], [missile_z[0]], color='blue', s=100, marker='o', label='导弹起点')
    ax.scatter([missile_x[-1]], [missile_y[-1]], [missile_z[-1]], color='blue', s=100, marker='s', label='导弹终点')
    ax.scatter([uav_x[0]], [uav_y[0]], [uav_z[0]], color='red', s=100, marker='o', label='无人机起点')
    ax.scatter([uav_x[-1]], [uav_y[-1]], [uav_z[-1]], color='red', s=100, marker='s', label='无人机终点')
    
    # 绘制目标点
    target_centers = [
        TARGETS["true_target"]["base_center"],
        [TARGETS["true_target"]["base_center"][0], 
         TARGETS["true_target"]["base_center"][1], 
         TARGETS["true_target"]["base_center"][2] + TARGETS["true_target"]["height"]]
    ]
    
    # 绘制目标中心
    for i, center in enumerate(target_centers):
        ax.scatter([center[0]], [center[1]], [center[2]], 
                  color='green', s=80, marker='^', label=f'目标中心{i+1}')
    
    # 绘制虚假目标
    fake_target = TARGETS["fake_target"]
    ax.scatter([fake_target[0]], [fake_target[1]], [fake_target[2]], 
              color='orange', s=120, marker='*', label='虚假目标')
    
    # 设置坐标轴
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'导弹与无人机轨迹可视化\n无人机方向: {uav_direction}°, 速度: {uav_speed}m/s')
    ax.legend()
    
    # 设置视角
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    # 打印轨迹信息
    print(f"\n=== 轨迹验证信息 ===")
    print(f"导弹{missile_id}:")
    print(f"  起点: {missile_points[0]}")
    print(f"  终点: {missile_points[-1]}")
    print(f"  距离原点: 起点{np.linalg.norm(missile_points[0]):.1f}m -> 终点{np.linalg.norm(missile_points[-1]):.1f}m")
    
    print(f"\n无人机{uav_id} (方向{uav_direction}°, 速度{uav_speed}m/s):")
    print(f"  起点: {uav_points[0]}")
    print(f"  终点: {uav_points[-1]}")
    print(f"  高度变化: {uav_points[0][2]:.1f}m -> {uav_points[-1][2]:.1f}m")


def visualize_smoke_masking(
    uav_direction: float = 90,
    uav_speed: float = 120,
    smoke_deploy_time: float = 5.0,
    smoke_explode_delay: float = 1.5,
    time_range: tuple = None
):
    """
    可视化烟幕遮蔽效果
    
    Args:
        uav_direction: 无人机飞行方向（度）
        uav_speed: 无人机速度
        smoke_deploy_time: 烟幕弹投放时间
        smoke_explode_delay: 起爆延时
        time_range: 时间范围，如果为None则自动计算
    """
    traj_calc = TrajectoryCalculator()
    
    # 创建轨迹函数
    missile_traj = traj_calc.create_missile_trajectory("M1")
    uav_traj = traj_calc.create_uav_trajectory("FY1", uav_direction, uav_speed)
    smoke_traj = traj_calc.create_smoke_trajectory(uav_traj, smoke_deploy_time, smoke_explode_delay)
    
    # 设置时间范围
    if time_range is None:
        explode_time = smoke_deploy_time + smoke_explode_delay
        time_range = (explode_time, explode_time + 5)  # 起爆后5秒
    
    t_start, t_end = time_range
    times = np.linspace(t_start, t_end, 50)
    
    # 计算轨迹点
    missile_points = [missile_traj(t) for t in times]
    uav_points = [uav_traj(t) for t in times]
    smoke_points = [smoke_traj(t) for t in times]
    
    # 创建3D图
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取坐标
    missile_x = [p[0] for p in missile_points]
    missile_y = [p[1] for p in missile_points]
    missile_z = [p[2] for p in missile_points]
    
    uav_x = [p[0] for p in uav_points]
    uav_y = [p[1] for p in uav_points]
    uav_z = [p[2] for p in uav_points]
    
    smoke_x = [p[0] for p in smoke_points]
    smoke_y = [p[1] for p in smoke_points]
    smoke_z = [p[2] for p in smoke_points]
    
    # 绘制轨迹
    ax.plot(missile_x, missile_y, missile_z, 'b-', linewidth=2, label='导弹轨迹')
    ax.plot(uav_x, uav_y, uav_z, 'r-', linewidth=2, label='无人机轨迹')
    ax.plot(smoke_x, smoke_y, smoke_z, 'g-', linewidth=2, label='烟幕云轨迹')
    
    # 标记关键点
    ax.scatter([missile_x[0]], [missile_y[0]], [missile_z[0]], color='blue', s=100, marker='o', label='导弹起始位置')
    ax.scatter([smoke_x[0]], [smoke_y[0]], [smoke_z[0]], color='green', s=100, marker='o', label='烟幕云起爆位置')
    
    # 绘制目标中心
    target_centers = [
        TARGETS["true_target"]["base_center"],
        [TARGETS["true_target"]["base_center"][0], 
         TARGETS["true_target"]["base_center"][1], 
         TARGETS["true_target"]["base_center"][2] + TARGETS["true_target"]["height"]]
    ]
    
    for i, center in enumerate(target_centers):
        ax.scatter([center[0]], [center[1]], [center[2]], 
                  color='purple', s=80, marker='s', label=f'目标中心{i+1}')
    
    # 绘制虚假目标
    fake_target = TARGETS["fake_target"]
    ax.scatter([fake_target[0]], [fake_target[1]], [fake_target[2]], 
              color='orange', s=120, marker='*', label='虚假目标')
    
    # 在几个关键时间点绘制目标点和视线
    sample_times = [times[0], times[len(times)//2], times[-1]]
    colors = ['cyan', 'magenta', 'yellow']
    
    for i, t in enumerate(sample_times):
        missile_pos = missile_traj(t)
        
        # 计算并绘制目标点
        for j, center in enumerate(target_centers):
            target_points = get_top_plane_points(missile_pos, center, TARGETS["true_target"]["radius"])
            
            target_x = [p[0] for p in target_points]
            target_y = [p[1] for p in target_points]
            target_z = [p[2] for p in target_points]
            
            ax.scatter(target_x, target_y, target_z, 
                      color=colors[i], s=30, alpha=0.6, 
                      label=f't={t:.1f}s目标点' if j == 0 else "")
            
            # 绘制视线
            for target_pos in target_points:
                ax.plot([missile_pos[0], target_pos[0]], 
                       [missile_pos[1], target_pos[1]], 
                       [missile_pos[2], target_pos[2]], 
                       color=colors[i], alpha=0.3, linewidth=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'烟幕遮蔽效果可视化\n投放时间: {smoke_deploy_time}s, 起爆延时: {smoke_explode_delay}s')
    ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 打印详细信息
    print(f"\n=== 烟幕遮蔽验证信息 ===")
    print(f"烟幕弹投放时间: {smoke_deploy_time}s")
    print(f"起爆延时: {smoke_explode_delay}s")
    print(f"起爆时间: {smoke_deploy_time + smoke_explode_delay}s")
    print(f"起爆位置: {smoke_traj(smoke_deploy_time + smoke_explode_delay)}")
    
    # 检查几个时间点的遮蔽情况
    for t in sample_times:
        missile_pos = missile_traj(t)
        smoke_pos = smoke_traj(t)
        print(f"\n时间 t={t:.1f}s:")
        print(f"  导弹位置: ({missile_pos[0]:.1f}, {missile_pos[1]:.1f}, {missile_pos[2]:.1f})")
        print(f"  烟幕云位置: ({smoke_pos[0]:.1f}, {smoke_pos[1]:.1f}, {smoke_pos[2]:.1f})")
        print(f"  导弹-烟幕云距离: {np.linalg.norm([missile_pos[i] - smoke_pos[i] for i in range(3)]):.1f}m")


if __name__ == "__main__":
    print("开始轨迹可视化验证...")
    
    # 基础轨迹可视化
    print("\n1. 基础轨迹验证:")
    visualize_trajectories()
    
    # 烟幕遮蔽可视化
    print("\n2. 烟幕遮蔽效果验证:")
    visualize_smoke_masking()
