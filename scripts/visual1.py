"""
问题1可视化脚本 - 生成三张必需图表

本脚本生成以下三张图：
1. 必做图1：三维静态轨迹图
2. 必做图2：遮蔽效果时序图 (Gantt Chart)
3. 加分图3：关键量变化图

功能：
- 基于Problem1.py的参数和逻辑生成静态图表
- 保存高质量图像文件
- 提供清晰的标注和说明
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import os
from datetime import datetime

# 导入求解器模块
import sys
sys.path.append('..')
from solver import (
    calculate_single_uav_single_smoke_masking,
    CONSTANTS, TARGETS, MISSILES, UAVS, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator
from solver.geometry import get_top_plane_points, get_under_points, distance_between, feet_of_perpendicular_to_anchor_target_lines

# 配置matplotlib字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']  # 中文字体优先级
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体

# 检查并设置中文字体
import matplotlib.font_manager as fm

def setup_fonts():
    """设置字体配置"""
    # 获取可用字体列表
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 设置中文字体优先级
    chinese_fonts = []
    for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        if font_name in available_fonts:
            chinese_fonts.append(font_name)
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
        print(f"使用中文字体: {chinese_fonts[0]}")
    else:
        # 如果没有找到标准中文字体，尝试其他可能的中文字体
        fallback_fonts = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['yahei', 'hei', 'sun', 'kai', 'fangsong'])]
        if fallback_fonts:
            plt.rcParams['font.sans-serif'] = fallback_fonts[:1] + ['DejaVu Sans']
            print(f"使用备用中文字体: {fallback_fonts[0]}")
        else:
            print("警告: 未找到合适的中文字体，可能影响中文显示")

# 设置字体
setup_fonts()


class Problem1Visualizer:
    """问题1可视化器"""
    
    def __init__(self, 
                 uav_direction=None,
                 uav_speed=120.0,
                 smoke_deploy_time=1.5,
                 smoke_explode_delay=3.6,
                 max_time=50.0):
        
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
        self.times = np.linspace(0, max_time, 1000)
        self.missile_pts = np.array([self.missile_traj(t) for t in self.times])
        self.uav_pts = np.array([self.uav_traj(t) for t in self.times])
        self.smoke_pts = np.array([self.smoke_traj(t) for t in self.times])
        
        # 计算遮蔽时间段
        self.masking_intervals = self._calculate_masking_intervals()
        
        # 创建输出目录
        self.output_dir = "output_figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _calculate_masking_intervals(self):
        """计算有效遮蔽时间段"""
        intervals = []
        time_step = 0.1
        times = np.arange(self.explode_time, min(self.max_time, self.explode_time + SMOKE_PARAMS["duration"]), time_step)
        
        is_masked = []
        for t in times:
            masked = self._is_masked_at_time(t)
            is_masked.append(masked)
        
        # 找到连续的遮蔽区间
        in_interval = False
        start_time = None
        
        for i, (t, masked) in enumerate(zip(times, is_masked)):
            if masked and not in_interval:
                # 开始新的遮蔽区间
                start_time = t
                in_interval = True
            elif not masked and in_interval:
                # 结束当前遮蔽区间
                intervals.append((start_time, t))
                in_interval = False
        
        # 处理最后一个区间
        if in_interval and start_time is not None:
            intervals.append((start_time, times[-1]))
        
        return intervals
    
    def _is_masked_at_time(self, t):
        """判断时间t是否被有效遮蔽"""
        if t < self.explode_time:
            return False
        
        # 获取当前时刻的导弹和烟幕云位置
        missile_pos = self.missile_traj(t)
        smoke_pos = self.smoke_traj(t)
        
        # 计算目标点
        true_center = TARGETS["true_target"]["base_center"]
        true_height = TARGETS["true_target"]["height"]
        true_radius = TARGETS["true_target"]["radius"]
        
        top_center = [true_center[0], true_center[1], true_center[2] + true_height]
        
        top_points = get_top_plane_points(missile_pos, top_center, true_radius)
        under_points = get_under_points(missile_pos, true_center, true_radius)
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
        return max(distances) <= SMOKE_PARAMS["cloud_radius"] if distances else False
    
    def _calculate_distance_to_sight_line(self, t):
        """计算烟幕球心到导弹视线的最小距离"""
        if t < self.explode_time:
            return float('inf')
        
        missile_pos = self.missile_traj(t)
        smoke_pos = self.smoke_traj(t)
        
        # 计算目标点
        true_center = TARGETS["true_target"]["base_center"]
        true_height = TARGETS["true_target"]["height"]
        true_radius = TARGETS["true_target"]["radius"]
        
        top_center = [true_center[0], true_center[1], true_center[2] + true_height]
        
        top_points = get_top_plane_points(missile_pos, top_center, true_radius)
        under_points = get_under_points(missile_pos, true_center, true_radius)
        all_target_points = top_points + under_points
        
        # 计算烟幕云到各条视线的距离
        feet = feet_of_perpendicular_to_anchor_target_lines(
            smoke_pos, missile_pos, all_target_points
        )
        
        distances = []
        for foot in feet:
            dist = distance_between(smoke_pos, foot)
            distances.append(dist)
        
        return max(distances) if distances else float('inf')
    
    def draw_cylinder(self, ax, center, radius, height, color='purple', alpha=0.3):
        """绘制圆柱体表示真目标"""
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
    
    def plot_3d_trajectory(self):
        """必做图1：三维静态轨迹图"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制完整轨迹
        ax.plot(self.missile_pts[:, 0], self.missile_pts[:, 1], 
                self.missile_pts[:, 2], 'b-', linewidth=2, label='导弹M1直线轨迹')
        ax.plot(self.uav_pts[:, 0], self.uav_pts[:, 1], 
                self.uav_pts[:, 2], 'r-', linewidth=2, label='无人机FY1直线轨迹')
        
        # 烟幕弹轨迹（分段绘制：抛物线 + 下沉）
        explode_idx = int(self.explode_time / self.max_time * len(self.times))
        ax.plot(self.smoke_pts[:explode_idx, 0], self.smoke_pts[:explode_idx, 1], 
                self.smoke_pts[:explode_idx, 2], 'g--', linewidth=2, label='烟幕弹抛物线轨迹')
        ax.plot(self.smoke_pts[explode_idx:, 0], self.smoke_pts[explode_idx:, 1], 
                self.smoke_pts[explode_idx:, 2], 'orange', linestyle='-', linewidth=2, label='烟幕云团下沉轨迹')
        
        # 绘制虚假目标（原点）
        fake_target = TARGETS["fake_target"]
        ax.scatter([fake_target[0]], [fake_target[1]], [fake_target[2]], 
                   c='gold', s=300, marker='*', label='虚假目标', edgecolors='black', linewidth=2)
        
        # 绘制真目标（圆柱体）
        true_center = TARGETS["true_target"]["base_center"]
        true_radius = TARGETS["true_target"]["radius"]
        true_height = TARGETS["true_target"]["height"]
        self.draw_cylinder(ax, true_center, true_radius, true_height, 
                          color='purple', alpha=0.4)
        
        # 添加真目标标签
        ax.text(true_center[0], true_center[1], true_center[2] + true_height + 20,
                '真目标', fontsize=12, color='purple', weight='bold')
        
        # 标记关键点
        # 起始位置
        ax.scatter([MISSILES["M1"]["initial_pos"][0]], [MISSILES["M1"]["initial_pos"][1]], 
                   [MISSILES["M1"]["initial_pos"][2]], c='blue', s=100, marker='o', label='导弹起始位置')
        ax.scatter([UAVS["FY1"]["initial_pos"][0]], [UAVS["FY1"]["initial_pos"][1]], 
                   [UAVS["FY1"]["initial_pos"][2]], c='red', s=100, marker='o', label='无人机起始位置')
        
        # 烟幕弹投放点
        deploy_pos = self.uav_traj(self.smoke_deploy_time)
        ax.scatter([deploy_pos[0]], [deploy_pos[1]], [deploy_pos[2]], 
                   c='green', s=150, marker='^', label='烟幕弹投放点')
        
        # 烟幕弹起爆点
        explode_pos = self.smoke_traj(self.explode_time)
        ax.scatter([explode_pos[0]], [explode_pos[1]], [explode_pos[2]], 
                   c='orange', s=200, marker='X', label='烟幕弹起爆点')
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('问题1 - 三维静态轨迹图', fontsize=16, weight='bold')
        
        # 设置视角
        ax.view_init(elev=25, azim=60)
        
        # 设置坐标范围 - 调整比例突出投放和起爆点区域
        # 重点关注无人机和烟幕弹活动区域
        uav_x_range = [self.uav_pts[:, 0].min(), self.uav_pts[:, 0].max()]
        uav_y_range = [self.uav_pts[:, 1].min(), self.uav_pts[:, 1].max()]
        smoke_x_range = [self.smoke_pts[:, 0].min(), self.smoke_pts[:, 0].max()]
        smoke_y_range = [self.smoke_pts[:, 1].min(), self.smoke_pts[:, 1].max()]
        
        # 扩大关键区域的显示范围
        focus_x_min = min(uav_x_range[0], smoke_x_range[0]) - 2000
        focus_x_max = max(uav_x_range[1], smoke_x_range[1]) + 2000
        focus_y_min = min(uav_y_range[0], smoke_y_range[0]) - 1000
        focus_y_max = max(uav_y_range[1], smoke_y_range[1]) + 1000
        
        # 保持导弹轨迹的可见性，但重点突出关键区域
        ax.set_xlim([max(focus_x_min, 5000), min(focus_x_max, 20000)])
        ax.set_ylim([focus_y_min, focus_y_max])
        ax.set_zlim([0, 2500])
        
        # 调整坐标轴刻度间距以减少密集感
        ax.set_xticks(np.arange(6000, 20000, 2000))
        ax.set_yticks(np.arange(-500, 1000, 250))
        ax.set_zticks(np.arange(0, 2500, 500))
        
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"图1_三维静态轨迹图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图1已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_masking_gantt(self):
        """必做图2：遮蔽效果时序图 (Gantt Chart)"""
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # 时间轴设置
        smoke_end_time = min(self.explode_time + SMOKE_PARAMS["duration"], self.max_time)
        time_range = np.linspace(0, smoke_end_time, int(smoke_end_time * 10))
        
        # 绘制背景时间轴
        ax.barh(0, smoke_end_time, height=0.8, color='lightgray', alpha=0.3, label='总时间范围')
        
        # 绘制烟幕弹各阶段
        # 等待投放阶段
        ax.barh(0, self.smoke_deploy_time, height=0.6, color='gray', alpha=0.6, label='等待投放')
        
        # 烟幕弹飞行阶段
        flight_duration = self.smoke_explode_delay
        ax.barh(0, flight_duration, left=self.smoke_deploy_time, height=0.6, 
                color='yellow', alpha=0.7, label='烟幕弹飞行')
        
        # 烟幕云生效阶段
        smoke_duration = min(SMOKE_PARAMS["duration"], smoke_end_time - self.explode_time)
        ax.barh(0, smoke_duration, left=self.explode_time, height=0.6, 
                color='green', alpha=0.4, label='烟幕云生效')
        
        # 绘制有效遮蔽时间段
        total_masking_duration = 0
        for i, (start, end) in enumerate(self.masking_intervals):
            duration = end - start
            total_masking_duration += duration
            ax.barh(0, duration, left=start, height=0.4, 
                    color='red', alpha=0.8, 
                    label='有效遮蔽' if i == 0 else "")
        
        # 标记关键时间点
        ax.axvline(x=self.smoke_deploy_time, color='blue', linestyle='--', alpha=0.7)
        ax.text(self.smoke_deploy_time, 0.6, f'投放\n{self.smoke_deploy_time}s', 
                ha='center', va='bottom', fontsize=10, weight='bold')
        
        ax.axvline(x=self.explode_time, color='orange', linestyle='--', alpha=0.7)
        ax.text(self.explode_time, 0.6, f'起爆\n{self.explode_time}s', 
                ha='center', va='bottom', fontsize=10, weight='bold')
        
        # 设置坐标轴
        ax.set_xlim(0, smoke_end_time)
        ax.set_ylim(-0.5, 1)
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('')
        ax.set_yticks([])
        ax.set_title('问题1 - 遮蔽效果时序图 (Gantt Chart)', fontsize=16, weight='bold')
        
        # 添加网格
        ax.grid(True, axis='x', alpha=0.3)
        
        # 添加图例
        ax.legend(loc='upper right')
        
        # 添加统计信息
        info_text = f"总有效遮蔽时长: {total_masking_duration:.2f}秒\n"
        info_text += f"遮蔽区间数: {len(self.masking_intervals)}\n"
        if self.masking_intervals:
            info_text += f"遮蔽区间: {[f'[{s:.2f}, {e:.2f}]' for s, e in self.masking_intervals]}"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"图2_遮蔽效果时序图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图2已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_distance_variation(self):
        """加分图3：关键量变化图"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 计算距离变化曲线 - 只显示遮蔽区间及周围部分
        if not self.masking_intervals:
            print("没有遮蔽区间，无法生成图3")
            return None
            
        # 确定显示时间范围：遮蔽区间前后各扩展2秒
        all_starts = [start for start, end in self.masking_intervals]
        all_ends = [end for start, end in self.masking_intervals]
        display_start = max(self.explode_time, min(all_starts) - 0.1)
        display_end = min(self.explode_time + SMOKE_PARAMS["duration"], max(all_ends) + 0.1)
        
        times = np.linspace(display_start, display_end, 500)
        distances = []
        
        for t in times:
            dist = self._calculate_distance_to_sight_line(t)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # 绘制距离曲线
        ax.plot(times, distances, 'b-', linewidth=2, label='烟幕球心到导弹视线距离')
        
        # 绘制有效半径水平线
        effective_radius = SMOKE_PARAMS["cloud_radius"]
        ax.axhline(y=effective_radius, color='red', linestyle='--', linewidth=2, 
                   label=f'烟幕有效半径 ({effective_radius}m)')
        
        # 标记有效遮蔽区域
        for i, (start, end) in enumerate(self.masking_intervals):
            if start >= display_start and end <= display_end:
                # 找到对应的距离值
                start_idx = np.argmin(np.abs(times - start))
                end_idx = np.argmin(np.abs(times - end))
                
                ax.fill_between(times[start_idx:end_idx+1], 
                               distances[start_idx:end_idx+1], 
                               effective_radius,
                               where=(distances[start_idx:end_idx+1] <= effective_radius),
                               alpha=0.3, color='green',
                               label='有效遮蔽区间' if i == 0 else "")
        
        # 设置坐标轴
        ax.set_xlabel('时间 (s)', fontsize=12)
        ax.set_ylabel('距离 (m)', fontsize=12)
        ax.set_title('问题1 - 导弹视线与烟幕球心距离变化图', fontsize=16, weight='bold')
        
        # 设置显示范围
        ax.set_xlim(display_start, display_end)
        
        # 设置y轴范围
        max_dist = np.max(distances[np.isfinite(distances)])
        ax.set_ylim(0, max(max_dist * 1.1, effective_radius * 2))
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        # 添加图例
        ax.legend()
        
        # 添加统计信息
        min_distance = np.min(distances[np.isfinite(distances)])
        avg_distance = np.mean(distances[np.isfinite(distances)])
        
        info_text = f"最小距离: {min_distance:.2f}m\n"
        info_text += f"平均距离: {avg_distance:.2f}m\n"
        info_text += f"有效半径: {effective_radius}m\n"
        info_text += f"遮蔽时长: {sum(end - start for start, end in self.masking_intervals):.2f}s"
        
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"图3_关键量变化图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"图3已保存: {filepath}")
        
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """生成所有三张图"""
        print("=" * 60)
        print("问题1可视化 - 生成三张图表")
        print("=" * 60)
        
        print(f"参数设置：")
        print(f"  无人机速度: {self.uav_speed} m/s")
        print(f"  烟幕弹投放时间: {self.smoke_deploy_time} s")
        print(f"  烟幕弹起爆延时: {self.smoke_explode_delay} s")
        print(f"  起爆时间: {self.explode_time} s")
        print(f"  有效遮蔽时长: {sum(end - start for start, end in self.masking_intervals):.2f} s")
        print()
        
        # 生成三张图
        print("正在生成图1：三维静态轨迹图...")
        self.plot_3d_trajectory()
        
        print("正在生成图2：遮蔽效果时序图...")
        self.plot_masking_gantt()
        
        print("正在生成图3：关键量变化图...")
        self.plot_distance_variation()
        
        print(f"\n所有图表已保存到 {self.output_dir} 文件夹")
        print("生成完成！")


def main():
    """主函数"""
    # 参数设置（与Problem1.py保持一致）
    uav_direction = None  # None表示自动计算朝向虚假目标
    uav_speed = 120.0
    smoke_deploy_time = 1.5
    smoke_explode_delay = 3.6
    
    # 创建可视化器并生成所有图表
    visualizer = Problem1Visualizer(
        uav_direction=uav_direction,
        uav_speed=uav_speed,
        smoke_deploy_time=smoke_deploy_time,
        smoke_explode_delay=smoke_explode_delay,
        max_time=25.0  # 减少计算时间
    )
    
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main() 