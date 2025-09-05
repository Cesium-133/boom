"""
问题1：单无人机、单烟幕弹、单导弹的有效遮蔽时长计算与可视化

功能：
1. 计算有效遮蔽时长
2. 交互式3D可视化，包含时间轴控制和动画播放
3. 显示真假目标（真目标用圆柱表示）
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 使用本地包
from solver import (
    calculate_single_uav_single_smoke_masking,
    CONSTANTS, TARGETS, MISSILES, UAVS, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator
from solver.geometry import get_top_plane_points, get_under_points

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def solve_problem1(
    uav_direction: float = None,  # None表示自动计算朝向虚假目标
    uav_speed: float = 120.0,
    smoke_deploy_time: float = 1.5,
    smoke_explode_delay: float = 3.6
):
    """
    问题1：单无人机、单烟幕弹、单导弹的有效遮蔽时长计算
    
    Args:
        uav_direction: 无人机飞行方向（度），None表示自动朝向虚假目标
        uav_speed: 无人机速度（m/s）
        smoke_deploy_time: 烟幕弹投放时间（s）
        smoke_explode_delay: 起爆延时（s）
    
    Returns:
        有效遮蔽时长（秒）
    """
    # 如果没有指定方向，自动计算朝向虚假目标的方向
    if uav_direction is None:
        uav_pos = UAVS["FY1"]["initial_pos"]
        fake_target = TARGETS["fake_target"]
        dx = fake_target[0] - uav_pos[0]
        dy = fake_target[1] - uav_pos[1]
        import math
        uav_direction = math.degrees(math.atan2(dy, dx))
    
    duration = calculate_single_uav_single_smoke_masking(
        uav_direction=uav_direction,
        uav_speed=uav_speed,
        smoke_deploy_time=smoke_deploy_time,
        smoke_explode_delay=smoke_explode_delay,
    )
    return duration


def draw_cylinder(ax, center, radius, height, color='purple', alpha=0.3):
    """绘制圆柱体表示真目标"""
    # 圆柱体参数
    theta = np.linspace(0, 2*np.pi, 30)
    z = np.array([0, height])
    
    # 生成圆柱体表面
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = center[0] + radius * np.cos(theta_grid)
    y_grid = center[1] + radius * np.sin(theta_grid)
    z_grid = center[2] + z_grid
    
    # 绘制圆柱体侧面
    ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=True)
    
    # 绘制顶部和底部圆
    for z_val in [center[2], center[2] + height]:
        x_circle = center[0] + radius * np.cos(theta)
        y_circle = center[1] + radius * np.sin(theta)
        z_circle = np.full_like(x_circle, z_val)
        ax.plot(x_circle, y_circle, z_circle, color=color, linewidth=2)


class InteractiveVisualizer:
    """交互式可视化器"""
    
    def __init__(self, 
                 uav_direction=None,
                 uav_speed=120.0,
                 smoke_deploy_time=1.5,
                 smoke_explode_delay=3.6,
                 max_time=20.0):
        
        self.uav_direction = uav_direction
        self.uav_speed = uav_speed
        self.smoke_deploy_time = smoke_deploy_time
        self.smoke_explode_delay = smoke_explode_delay
        self.max_time = max_time
        
        # 创建轨迹计算器
        self.traj = TrajectoryCalculator()
        self.missile_traj = self.traj.create_missile_trajectory("M1")
        self.uav_traj = self.traj.create_uav_trajectory("FY1", 
                                                        direction_degrees=uav_direction, 
                                                        speed=uav_speed)
        self.smoke_traj = self.traj.create_smoke_trajectory(self.uav_traj, 
                                                           smoke_deploy_time, 
                                                           smoke_explode_delay)
        
        # 计算关键时间点
        self.explode_time = smoke_deploy_time + smoke_explode_delay
        
        # 预计算轨迹数据
        self.times = np.linspace(0, max_time, 500)
        self.missile_pts = np.array([self.missile_traj(t) for t in self.times])
        self.uav_pts = np.array([self.uav_traj(t) for t in self.times])
        self.smoke_pts = np.array([self.smoke_traj(t) for t in self.times])
        
        # 动画相关
        self.is_playing = False
        self.animation = None
        
        # 创建图形
        self.setup_figure()
        
    def setup_figure(self):
        """设置图形界面"""
        self.fig = plt.figure(figsize=(14, 10))
        
        # 3D轴
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 调整3D图的位置，为控件留出空间
        self.ax.set_position([0.1, 0.25, 0.8, 0.65])
        
        # 时间滑块
        ax_time = plt.axes([0.15, 0.12, 0.65, 0.03])
        self.time_slider = Slider(ax_time, '时间 (s)', 0, self.max_time, 
                                  valinit=0, valstep=0.1)
        self.time_slider.on_changed(self.update_time)
        
        # 播放/暂停按钮
        ax_play = plt.axes([0.85, 0.12, 0.1, 0.04])
        self.play_button = Button(ax_play, '播放')
        self.play_button.on_clicked(self.toggle_play)
        
        # 速度滑块
        ax_speed = plt.axes([0.15, 0.07, 0.65, 0.03])
        self.speed_slider = Slider(ax_speed, '播放速度', 0.1, 5.0, 
                                  valinit=1.0, valstep=0.1)
        
        # 初始化绘图
        self.init_plot()
        
    def init_plot(self):
        """初始化绘图元素"""
        self.ax.clear()
        
        # 绘制完整轨迹（细线）
        self.ax.plot(self.missile_pts[:, 0], self.missile_pts[:, 1], 
                    self.missile_pts[:, 2], 'b-', alpha=0.3, linewidth=1, label='导弹轨迹')
        self.ax.plot(self.uav_pts[:, 0], self.uav_pts[:, 1], 
                    self.uav_pts[:, 2], 'r-', alpha=0.3, linewidth=1, label='无人机轨迹')
        self.ax.plot(self.smoke_pts[:, 0], self.smoke_pts[:, 1], 
                    self.smoke_pts[:, 2], 'g-', alpha=0.3, linewidth=1, label='烟幕轨迹')
        
        # 绘制虚假目标（原点）
        fake_target = TARGETS["fake_target"]
        self.ax.scatter([fake_target[0]], [fake_target[1]], [fake_target[2]], 
                       c='orange', s=200, marker='*', label='虚假目标', edgecolors='black', linewidth=2)
        
        # 绘制真目标（圆柱体）
        true_center = TARGETS["true_target"]["base_center"]
        true_radius = TARGETS["true_target"]["radius"]
        true_height = TARGETS["true_target"]["height"]
        draw_cylinder(self.ax, true_center, true_radius, true_height, 
                     color='purple', alpha=0.3)
        
        # 添加真目标标签
        self.ax.text(true_center[0], true_center[1], true_center[2] + true_height + 5,
                    '真目标', fontsize=10, color='purple')
        
        # 当前位置点（将在update中更新）
        self.missile_point, = self.ax.plot([], [], [], 'bo', markersize=8, label='导弹')
        self.uav_point, = self.ax.plot([], [], [], 'ro', markersize=8, label='无人机')
        self.smoke_point, = self.ax.plot([], [], [], 'go', markersize=10, label='烟幕云')
        
        # 烟幕云球体（半透明）
        self.smoke_sphere = None
        
        # 视线
        self.sight_lines = []
        
        # 设置坐标轴
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('问题1 - 导弹拦截烟幕遮蔽可视化')
        
        # 设置视角
        self.ax.view_init(elev=20, azim=45)
        
        # 设置坐标范围
        self.set_axis_limits()
        
        self.ax.legend(loc='upper right')
        
    def set_axis_limits(self):
        """设置坐标轴范围"""
        # 计算所有点的范围
        all_x = np.concatenate([self.missile_pts[:, 0], self.uav_pts[:, 0], self.smoke_pts[:, 0]])
        all_y = np.concatenate([self.missile_pts[:, 1], self.uav_pts[:, 1], self.smoke_pts[:, 1]])
        all_z = np.concatenate([self.missile_pts[:, 2], self.uav_pts[:, 2], self.smoke_pts[:, 2]])
        
        # 添加一些边距
        margin = 500
        self.ax.set_xlim([min(all_x.min(), -margin), max(all_x.max(), margin)])
        self.ax.set_ylim([min(all_y.min(), -margin), max(all_y.max(), margin)])
        self.ax.set_zlim([0, max(all_z.max(), 2500)])
        
    def update_time(self, val):
        """更新时间显示"""
        t = self.time_slider.val
        
        # 获取当前位置
        missile_pos = self.missile_traj(t)
        uav_pos = self.uav_traj(t)
        smoke_pos = self.smoke_traj(t)
        
        # 更新点位置
        self.missile_point.set_data([missile_pos[0]], [missile_pos[1]])
        self.missile_point.set_3d_properties([missile_pos[2]])
        
        self.uav_point.set_data([uav_pos[0]], [uav_pos[1]])
        self.uav_point.set_3d_properties([uav_pos[2]])
        
        self.smoke_point.set_data([smoke_pos[0]], [smoke_pos[1]])
        self.smoke_point.set_3d_properties([smoke_pos[2]])
        
        # 更新烟幕云球体（如果已起爆）
        if self.smoke_sphere is not None:
            self.smoke_sphere.remove()
            self.smoke_sphere = None
            
        if t >= self.explode_time:
            # 绘制烟幕云球体
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 20)
            x = smoke_pos[0] + SMOKE_PARAMS["cloud_radius"] * np.outer(np.cos(u), np.sin(v))
            y = smoke_pos[1] + SMOKE_PARAMS["cloud_radius"] * np.outer(np.sin(u), np.sin(v))
            z = smoke_pos[2] + SMOKE_PARAMS["cloud_radius"] * np.outer(np.ones(np.size(u)), np.cos(v))
            self.smoke_sphere = self.ax.plot_surface(x, y, z, color='green', alpha=0.2)
        
        # 更新视线
        for line in self.sight_lines:
            line.remove()
        self.sight_lines = []
        
        # 绘制导弹到目标的视线
        true_center = TARGETS["true_target"]["base_center"]
        top_center = [true_center[0], true_center[1], true_center[2] + TARGETS["true_target"]["height"]]
        
        # 获取目标点
        top_points = get_top_plane_points(missile_pos, top_center, TARGETS["true_target"]["radius"])
        under_points = get_under_points(missile_pos, true_center, TARGETS["true_target"]["radius"])
        
        # 绘制视线（虚线）
        for point in top_points + under_points:
            line, = self.ax.plot([missile_pos[0], point[0]], 
                                [missile_pos[1], point[1]], 
                                [missile_pos[2], point[2]], 
                                'y--', alpha=0.3, linewidth=0.5)
            self.sight_lines.append(line)
        
        # 更新标题显示当前时间和状态
        status = ""
        if t < self.smoke_deploy_time:
            status = "等待投放"
        elif t < self.explode_time:
            status = "烟幕弹飞行中"
        else:
            status = "烟幕云生效"
            
        self.ax.set_title(f'问题1 - 时间: {t:.1f}s - 状态: {status}')
        
        self.fig.canvas.draw_idle()
        
    def toggle_play(self, event):
        """切换播放/暂停"""
        if self.is_playing:
            self.is_playing = False
            if self.animation is not None:
                self.animation.event_source.stop()
            self.play_button.label.set_text('播放')
        else:
            self.is_playing = True
            self.play_button.label.set_text('暂停')
            self.start_animation()
            
    def start_animation(self):
        """开始动画"""
        def animate(frame):
            if not self.is_playing:
                return
            
            # 更新时间
            current_time = self.time_slider.val
            speed = self.speed_slider.val
            new_time = current_time + 0.1 * speed
            
            if new_time > self.max_time:
                new_time = 0
                
            self.time_slider.set_val(new_time)
            
        self.animation = FuncAnimation(self.fig, animate, interval=50, blit=False)
        
    def show(self):
        """显示可视化界面"""
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("问题1：单无人机、单烟幕弹、单导弹的有效遮蔽时长计算")
    print("=" * 60)
    
    # 参数设置
    uav_direction = None  # None表示自动计算朝向虚假目标
    uav_speed = 120.0
    smoke_deploy_time = 1.5
    smoke_explode_delay = 3.6
    
    # 计算有效遮蔽时长
    duration = solve_problem1(
        uav_direction=uav_direction,
        uav_speed=uav_speed,
        smoke_deploy_time=smoke_deploy_time,
        smoke_explode_delay=smoke_explode_delay,
    )
    
    print(f"\n参数设置：")
    print(f"  无人机速度: {uav_speed} m/s")
    print(f"  无人机方向: {'自动朝向虚假目标' if uav_direction is None else f'{uav_direction}°'}")
    print(f"  烟幕弹投放时间: {smoke_deploy_time} s")
    print(f"  烟幕弹起爆延时: {smoke_explode_delay} s")
    print(f"  起爆时间: {smoke_deploy_time + smoke_explode_delay} s")
    
    print(f"\n计算结果：")
    print(f"  有效遮蔽时长: {duration:.2f} 秒")
    
    print("\n正在启动交互式可视化界面...")
    print("使用说明：")
    print("  - 拖动时间滑块查看不同时刻的状态")
    print("  - 点击'播放'按钮自动播放动画")
    print("  - 调整播放速度滑块改变动画速度")
    print("  - 鼠标拖动可旋转视角")
    
    # 创建交互式可视化
    visualizer = InteractiveVisualizer(
        uav_direction=uav_direction,
        uav_speed=uav_speed,
        smoke_deploy_time=smoke_deploy_time,
        smoke_explode_delay=smoke_explode_delay,
        max_time=50.0
    )
    
    visualizer.show()


if __name__ == "__main__":
    main()
