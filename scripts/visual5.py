"""
问题5可视化脚本 - 多无人机多导弹协同烟幕遮蔽优化结果可视化

本脚本生成以下图表：
1. 5架无人机与3枚导弹的三维轨迹图
2. 时间序列分析图（烟幕弹投放和起爆时间线）
3. 遮蔽效果分析图
4. 无人机配置雷达图

功能：
- 基于Problem5-DE-Multi-Missile.py的最优参数生成可视化
- 分析多无人机协同策略的时空配置
- 展示对3枚导弹的综合遮蔽效果
- 保存高质量图像文件
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import os
from datetime import datetime
import sys
from typing import Dict, List, Any

# 导入求解器模块
sys.path.append('..')
from solver import (
    calculate_problem5_multi_uav_multi_missile_masking,
    CONSTANTS, TARGETS, MISSILES, UAVS, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator
from solver.config import calculate_problem5_bounds

# 配置matplotlib字体显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 检查并设置中文字体
import matplotlib.font_manager as fm

def setup_fonts():
    """设置字体配置"""
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    
    chinese_fonts = []
    for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun']:
        if font_name in available_fonts:
            chinese_fonts.append(font_name)
    
    if chinese_fonts:
        plt.rcParams['font.sans-serif'] = chinese_fonts + ['DejaVu Sans']
        print(f"使用中文字体: {chinese_fonts[0]}")
    else:
        fallback_fonts = [f for f in available_fonts if any(keyword in f.lower() for keyword in ['yahei', 'hei', 'sun', 'kai', 'fangsong'])]
        if fallback_fonts:
            plt.rcParams['font.sans-serif'] = fallback_fonts[:1] + ['DejaVu Sans']
            print(f"使用备用中文字体: {fallback_fonts[0]}")
        else:
            print("警告: 未找到合适的中文字体，可能影响中文显示")

setup_fonts()


class Problem5Visualizer:
    """问题5可视化器"""
    
    def __init__(self, best_solution: Dict[str, Any] = None):
        """
        初始化可视化器
        
        Args:
            best_solution: 最优解决方案，包含5架无人机的参数
        """
        # 使用提供的最优解决方案
        if best_solution is None:
            # 默认使用用户提供的最优参数
            self.best_solution = {
                'uav_1_direction': 12.87, 'uav_1_speed': 138.39,
                'uav_2_direction': 118.25, 'uav_2_speed': 131.54,
                'uav_3_direction': 118.56, 'uav_3_speed': 83.05,
                'uav_4_direction': 249.41, 'uav_4_speed': 118.65,
                'uav_5_direction': 132.81, 'uav_5_speed': 98.85,
                # UAV1 烟幕弹
                'uav_1_smoke_1_deploy_time': 14.449, 'uav_1_smoke_1_explode_delay': 7.272,
                'uav_1_smoke_2_deploy_time': 55.251, 'uav_1_smoke_2_explode_delay': 6.252,
                'uav_1_smoke_3_deploy_time': 1.272, 'uav_1_smoke_3_explode_delay': 7.949,
                # UAV2 烟幕弹
                'uav_2_smoke_1_deploy_time': 0.217, 'uav_2_smoke_1_explode_delay': 5.580,
                'uav_2_smoke_2_deploy_time': 0.012, 'uav_2_smoke_2_explode_delay': 0.015,
                'uav_2_smoke_3_deploy_time': 0.015, 'uav_2_smoke_3_explode_delay': 0.598,
                # UAV3 烟幕弹
                'uav_3_smoke_1_deploy_time': 54.170, 'uav_3_smoke_1_explode_delay': 0.408,
                'uav_3_smoke_2_deploy_time': 14.748, 'uav_3_smoke_2_explode_delay': 0.332,
                'uav_3_smoke_3_deploy_time': 57.396, 'uav_3_smoke_3_explode_delay': 5.967,
                # UAV4 烟幕弹
                'uav_4_smoke_1_deploy_time': 64.056, 'uav_4_smoke_1_explode_delay': 5.756,
                'uav_4_smoke_2_deploy_time': 63.184, 'uav_4_smoke_2_explode_delay': 4.504,
                'uav_4_smoke_3_deploy_time': 7.453, 'uav_4_smoke_3_explode_delay': 6.280,
                # UAV5 烟幕弹
                'uav_5_smoke_1_deploy_time': 4.385, 'uav_5_smoke_1_explode_delay': 1.933,
                'uav_5_smoke_2_deploy_time': 12.337, 'uav_5_smoke_2_explode_delay': 1.558,
                'uav_5_smoke_3_deploy_time': 32.446, 'uav_5_smoke_3_explode_delay': 5.838
            }
            self.optimal_score = 10.451769
        else:
            self.best_solution = best_solution
            # 计算最优得分
            self.optimal_score = calculate_problem5_multi_uav_multi_missile_masking(**best_solution)
            
        self.max_time = 80.0  # 扩展时间范围以覆盖所有事件
        
        # 创建轨迹计算器
        self.traj = TrajectoryCalculator()
        
        # 创建3枚导弹的轨迹
        self.missile_trajs = {
            'M1': self.traj.create_missile_trajectory("M1"),
            'M2': self.traj.create_missile_trajectory("M2"),
            'M3': self.traj.create_missile_trajectory("M3")
        }
        
        # 创建5架无人机的轨迹
        self.uav_trajs = {}
        self.uav_names = ['FY1', 'FY2', 'FY3', 'FY4', 'FY5']
        for i in range(5):
            uav_id = i + 1
            uav_key = f'UAV{uav_id}'
            direction = self.best_solution[f'uav_{uav_id}_direction']
            speed = self.best_solution[f'uav_{uav_id}_speed']
            self.uav_trajs[uav_key] = self.traj.create_uav_trajectory(
                self.uav_names[i], direction_degrees=direction, speed=speed
            )
        
        # 创建所有烟幕弹轨迹
        self.smoke_trajs = {}
        for uav_id in range(1, 6):
            uav_key = f'UAV{uav_id}'
            for smoke_id in range(1, 4):
                smoke_key = f'{uav_key}_S{smoke_id}'
                deploy_time = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time']
                explode_delay = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay']
                
                smoke_traj = self.traj.create_smoke_trajectory(
                    self.uav_trajs[uav_key], deploy_time, explode_delay
                )
                self.smoke_trajs[smoke_key] = {
                    'trajectory': smoke_traj,
                    'deploy_time': deploy_time,
                    'explode_delay': explode_delay,
                    'explode_time': deploy_time + explode_delay,
                    'uav': uav_key
                }
        
        # 预计算轨迹数据
        self.times = np.linspace(0, self.max_time, 1600)
        
        # 计算导弹轨迹点
        self.missile_pts = {}
        for missile_id, traj in self.missile_trajs.items():
            self.missile_pts[missile_id] = np.array([traj(t) for t in self.times])
        
        # 计算无人机轨迹点
        self.uav_pts = {}
        for uav_key, traj in self.uav_trajs.items():
            self.uav_pts[uav_key] = np.array([traj(t) for t in self.times])
        
        # 创建输出目录
        self.output_dir = "output_figures"
        os.makedirs(self.output_dir, exist_ok=True)
    
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
    
    def plot_3d_multi_trajectory(self):
        """绘制5架无人机与3枚导弹的三维轨迹图"""
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # 导弹颜色
        missile_colors = ['blue', 'navy', 'darkblue']
        missile_styles = ['-', '--', '-.']
        
        # 绘制3枚导弹轨迹
        for i, (missile_id, pts) in enumerate(self.missile_pts.items()):
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                   color=missile_colors[i], linestyle=missile_styles[i], 
                   linewidth=2.5, label=f'导弹{missile_id}轨迹')
        
        # 无人机颜色
        uav_colors = ['red', 'orange', 'green', 'purple', 'brown']
        
        # 绘制5架无人机轨迹
        for i, (uav_key, pts) in enumerate(self.uav_pts.items()):
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 
                   color=uav_colors[i], linewidth=2, 
                   label=f'无人机{uav_key}轨迹')
        
        # 绘制虚假目标（原点）
        fake_target = TARGETS["fake_target"]
        ax.scatter([fake_target[0]], [fake_target[1]], [fake_target[2]], 
                   c='gold', s=400, marker='*', label='虚假目标', 
                   edgecolors='black', linewidth=2)
        
        # 绘制真目标（圆柱体）
        true_center = TARGETS["true_target"]["base_center"]
        true_radius = TARGETS["true_target"]["radius"]
        true_height = TARGETS["true_target"]["height"]
        self.draw_cylinder(ax, true_center, true_radius, true_height, 
                          color='purple', alpha=0.4)
        
        # 标记导弹起始位置
        for i, missile_id in enumerate(['M1', 'M2', 'M3']):
            pos = MISSILES[missile_id]["initial_pos"]
            ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                      c=missile_colors[i], s=120, marker='o', 
                      label=f'导弹{missile_id}起始位置')
        
        # 标记无人机起始位置
        for i, uav_name in enumerate(self.uav_names):
            pos = UAVS[uav_name]["initial_pos"]
            ax.scatter([pos[0]], [pos[1]], [pos[2]], 
                      c=uav_colors[i], s=120, marker='s', 
                      label=f'无人机{uav_name}起始位置')
        
        # 标记关键烟幕弹起爆点（选择几个重要的）
        important_smokes = [
            ('UAV2_S2', 'red'),    # 最早起爆
            ('UAV1_S3', 'orange'), # 早期重要
            ('UAV5_S1', 'green'),  # 中期重要
        ]
        
        for smoke_key, color in important_smokes:
            smoke_info = self.smoke_trajs[smoke_key]
            explode_pos = smoke_info['trajectory'](smoke_info['explode_time'])
            ax.scatter([explode_pos[0]], [explode_pos[1]], [explode_pos[2]], 
                      c=color, s=150, marker='X', alpha=0.8)
        
        # 设置坐标轴
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title('问题5 - 5架无人机对3枚导弹协同遮蔽三维轨迹图', fontsize=16, weight='bold')
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
        
        # 设置坐标范围
        ax.set_xlim([0, 20000])
        ax.set_ylim([-4000, 3000])
        ax.set_zlim([0, 2500])
        
        # 图例 - 分两列显示
        handles, labels = ax.get_legend_handles_labels()
        # 只显示主要轨迹，避免图例过于拥挤
        main_handles = handles[:8]  # 3个导弹 + 5个无人机
        main_labels = labels[:8]
        ax.legend(main_handles, main_labels, loc='upper right', 
                 bbox_to_anchor=(1, 1), ncol=2, fontsize=9)
        
        # 添加最优得分信息
        score_text = f"最优综合遮蔽时长: {self.optimal_score:.3f} 秒\n"
        score_text += f"使用无人机: 5架\n"
        score_text += f"使用烟幕弹: 15枚\n"
        score_text += f"针对导弹: M1, M2, M3"
        
        ax.text2D(0.02, 0.85, score_text, transform=ax.transAxes, 
                  fontsize=11, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题5_多无人机多导弹三维轨迹图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"三维轨迹图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_timeline_analysis(self):
        """绘制时间序列分析图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle('问题5 - 烟幕弹投放与起爆时间序列分析', fontsize=16, weight='bold')
        
        # 收集所有事件
        events = []
        uav_colors = ['red', 'orange', 'green', 'purple', 'brown']
        
        for uav_id in range(1, 6):
            color = uav_colors[uav_id - 1]
            for smoke_id in range(1, 4):
                deploy_time = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time']
                explode_delay = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay']
                explode_time = deploy_time + explode_delay
                
                events.append({
                    'uav': f'UAV{uav_id}',
                    'smoke': f'S{smoke_id}',
                    'deploy_time': deploy_time,
                    'explode_time': explode_time,
                    'color': color,
                    'uav_id': uav_id
                })
        
        # 排序事件
        events.sort(key=lambda x: x['deploy_time'])
        
        # 子图1: 投放时间线
        y_positions = []
        labels = []
        
        for i, event in enumerate(events):
            y_pos = event['uav_id'] - 1 + (int(event['smoke'][-1]) - 2) * 0.25  # 微调位置避免重叠
            y_positions.append(y_pos)
            labels.append(f"{event['uav']}-{event['smoke']}")
            
            # 投放时间点
            ax1.scatter(event['deploy_time'], y_pos, 
                       c=event['color'], s=80, marker='o', alpha=0.8)
            ax1.text(event['deploy_time'], y_pos + 0.1, 
                    f"{event['deploy_time']:.1f}s", 
                    fontsize=8, ha='center', va='bottom')
        
        ax1.set_xlabel('时间 (s)', fontsize=12)
        ax1.set_ylabel('无人机', fontsize=12)
        ax1.set_title('烟幕弹投放时间线', fontsize=14)
        ax1.set_yticks(range(5))
        ax1.set_yticklabels(['UAV1', 'UAV2', 'UAV3', 'UAV4', 'UAV5'])
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 70])
        
        # 子图2: 起爆时间线
        events.sort(key=lambda x: x['explode_time'])
        
        for i, event in enumerate(events):
            y_pos = event['uav_id'] - 1 + (int(event['smoke'][-1]) - 2) * 0.25
            
            # 起爆时间点
            ax2.scatter(event['explode_time'], y_pos, 
                       c=event['color'], s=80, marker='X', alpha=0.8)
            ax2.text(event['explode_time'], y_pos + 0.1, 
                    f"{event['explode_time']:.1f}s", 
                    fontsize=8, ha='center', va='bottom')
        
        ax2.set_xlabel('时间 (s)', fontsize=12)
        ax2.set_ylabel('无人机', fontsize=12)
        ax2.set_title('烟幕弹起爆时间线', fontsize=14)
        ax2.set_yticks(range(5))
        ax2.set_yticklabels(['UAV1', 'UAV2', 'UAV3', 'UAV4', 'UAV5'])
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim([0, 70])
        
        # 添加颜色图例
        legend_elements = [plt.scatter([], [], c=uav_colors[i], s=80, label=f'UAV{i+1}') 
                          for i in range(5)]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题5_时间序列分析图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"时间序列分析图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_uav_configuration_radar(self):
        """绘制无人机配置雷达图"""
        fig, axes = plt.subplots(1, 5, figsize=(20, 4), subplot_kw=dict(projection='polar'))
        fig.suptitle('问题5 - 无人机配置雷达图', fontsize=16, weight='bold')
        
        uav_colors = ['red', 'orange', 'green', 'purple', 'brown']
        
        for uav_id in range(1, 6):
            ax = axes[uav_id - 1]
            
            # 获取该无人机的参数
            direction = self.best_solution[f'uav_{uav_id}_direction']
            speed = self.best_solution[f'uav_{uav_id}_speed']
            
            # 烟幕弹时间分析
            deploy_times = []
            explode_delays = []
            for smoke_id in range(1, 4):
                deploy_times.append(self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time'])
                explode_delays.append(self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay'])
            
            # 雷达图数据（标准化到0-1）
            categories = ['飞行方向', '飞行速度', '平均投放时间', '平均延时', '时间分散度']
            values = [
                direction / 360.0,  # 方向标准化
                (speed - 70) / 70,  # 速度标准化（70-140范围）
                np.mean(deploy_times) / 70.0,  # 平均投放时间标准化
                np.mean(explode_delays) / 8.0,  # 平均延时标准化
                np.std(deploy_times) / 30.0     # 时间分散度标准化
            ]
            
            # 角度
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            values += values[:1]  # 闭合
            angles += angles[:1]
            
            # 绘制雷达图
            ax.plot(angles, values, 'o-', linewidth=2, color=uav_colors[uav_id - 1])
            ax.fill(angles, values, alpha=0.25, color=uav_colors[uav_id - 1])
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=8)
            ax.set_ylim(0, 1)
            ax.set_title(f'UAV{uav_id}', fontsize=12, weight='bold', 
                        color=uav_colors[uav_id - 1])
            ax.grid(True)
            
            # 添加具体数值标注
            param_text = f"方向: {direction:.1f}°\n"
            param_text += f"速度: {speed:.1f}m/s\n"
            param_text += f"投放: {np.mean(deploy_times):.1f}s\n"
            param_text += f"延时: {np.mean(explode_delays):.1f}s"
            
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题5_无人机配置雷达图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"无人机配置雷达图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_masking_effectiveness(self):
        """绘制遮蔽效果分析图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('问题5 - 遮蔽效果分析', fontsize=16, weight='bold')
        
        # 子图1: 烟幕弹投放时间分布
        all_deploy_times = []
        uav_labels = []
        uav_colors = ['red', 'orange', 'green', 'purple', 'brown']
        
        for uav_id in range(1, 6):
            for smoke_id in range(1, 4):
                deploy_time = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time']
                all_deploy_times.append(deploy_time)
                uav_labels.append(f'UAV{uav_id}')
        
        # 按UAV分组绘制投放时间分布
        for uav_id in range(1, 6):
            uav_times = []
            for smoke_id in range(1, 4):
                deploy_time = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time']
                uav_times.append(deploy_time)
            
            ax1.scatter([uav_id] * 3, uav_times, 
                       c=uav_colors[uav_id - 1], s=100, alpha=0.7, label=f'UAV{uav_id}')
        
        ax1.set_xlabel('无人机', fontsize=12)
        ax1.set_ylabel('投放时间 (s)', fontsize=12)
        ax1.set_title('各无人机烟幕弹投放时间分布', fontsize=14)
        ax1.set_xticks(range(1, 6))
        ax1.set_xticklabels([f'UAV{i}' for i in range(1, 6)])
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 子图2: 起爆延时分布
        for uav_id in range(1, 6):
            uav_delays = []
            for smoke_id in range(1, 4):
                explode_delay = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay']
                uav_delays.append(explode_delay)
            
            ax2.scatter([uav_id] * 3, uav_delays, 
                       c=uav_colors[uav_id - 1], s=100, alpha=0.7)
        
        ax2.set_xlabel('无人机', fontsize=12)
        ax2.set_ylabel('起爆延时 (s)', fontsize=12)
        ax2.set_title('各无人机烟幕弹起爆延时分布', fontsize=14)
        ax2.set_xticks(range(1, 6))
        ax2.set_xticklabels([f'UAV{i}' for i in range(1, 6)])
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 无人机速度和方向分析
        directions = []
        speeds = []
        for uav_id in range(1, 6):
            directions.append(self.best_solution[f'uav_{uav_id}_direction'])
            speeds.append(self.best_solution[f'uav_{uav_id}_speed'])
        
        scatter = ax3.scatter(directions, speeds, 
                             c=range(1, 6), s=200, alpha=0.7, 
                             cmap='viridis', edgecolors='black')
        
        # 添加标签
        for i, (dir_val, speed_val) in enumerate(zip(directions, speeds)):
            ax3.annotate(f'UAV{i+1}', (dir_val, speed_val), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax3.set_xlabel('飞行方向 (度)', fontsize=12)
        ax3.set_ylabel('飞行速度 (m/s)', fontsize=12)
        ax3.set_title('无人机速度-方向配置图', fontsize=14)
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax3, label='UAV编号')
        
        # 子图4: 时间覆盖分析
        time_bins = np.arange(0, 71, 5)  # 5秒间隔
        coverage_count = np.zeros(len(time_bins) - 1)
        
        # 统计每个时间段内活跃的烟幕弹数量
        for i in range(len(time_bins) - 1):
            t_start, t_end = time_bins[i], time_bins[i + 1]
            t_mid = (t_start + t_end) / 2
            
            active_count = 0
            for uav_id in range(1, 6):
                for smoke_id in range(1, 4):
                    deploy_time = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time']
                    explode_delay = self.best_solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay']
                    explode_time = deploy_time + explode_delay
                    
                    # 假设烟幕持续20秒
                    end_time = explode_time + 20
                    
                    if explode_time <= t_mid <= end_time:
                        active_count += 1
            
            coverage_count[i] = active_count
        
        ax4.bar(time_bins[:-1], coverage_count, width=4, alpha=0.7, 
               color='skyblue', edgecolor='navy')
        ax4.set_xlabel('时间 (s)', fontsize=12)
        ax4.set_ylabel('活跃烟幕云数量', fontsize=12)
        ax4.set_title('时间覆盖分析（烟幕云活跃度）', fontsize=14)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题5_遮蔽效果分析图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"遮蔽效果分析图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("=" * 80)
        print("问题5可视化 - 多无人机多导弹协同遮蔽优化结果")
        print("=" * 80)
        
        print(f"最优综合遮蔽时长: {self.optimal_score:.6f} 秒")
        print(f"使用无人机: 5架")
        print(f"使用烟幕弹: 15枚")
        print(f"针对导弹: M1, M2, M3")
        print()
        
        # 生成图表
        print("正在生成三维轨迹图...")
        self.plot_3d_multi_trajectory()
        
        print("正在生成时间序列分析图...")
        self.plot_timeline_analysis()
        
        print("正在生成无人机配置雷达图...")
        self.plot_uav_configuration_radar()
        
        print("正在生成遮蔽效果分析图...")
        self.plot_masking_effectiveness()
        
        print(f"\n所有图表已保存到 {self.output_dir} 文件夹")
        print("生成完成！")


def main():
    """主函数"""
    # 使用用户提供的最优解决方案
    best_solution = {
        'uav_1_direction': 12.87, 'uav_1_speed': 138.39,
        'uav_2_direction': 118.25, 'uav_2_speed': 131.54,
        'uav_3_direction': 118.56, 'uav_3_speed': 83.05,
        'uav_4_direction': 249.41, 'uav_4_speed': 118.65,
        'uav_5_direction': 132.81, 'uav_5_speed': 98.85,
        # UAV1 烟幕弹
        'uav_1_smoke_1_deploy_time': 14.449, 'uav_1_smoke_1_explode_delay': 7.272,
        'uav_1_smoke_2_deploy_time': 55.251, 'uav_1_smoke_2_explode_delay': 6.252,
        'uav_1_smoke_3_deploy_time': 1.272, 'uav_1_smoke_3_explode_delay': 7.949,
        # UAV2 烟幕弹
        'uav_2_smoke_1_deploy_time': 0.217, 'uav_2_smoke_1_explode_delay': 5.580,
        'uav_2_smoke_2_deploy_time': 0.012, 'uav_2_smoke_2_explode_delay': 0.015,
        'uav_2_smoke_3_deploy_time': 0.015, 'uav_2_smoke_3_explode_delay': 0.598,
        # UAV3 烟幕弹
        'uav_3_smoke_1_deploy_time': 54.170, 'uav_3_smoke_1_explode_delay': 0.408,
        'uav_3_smoke_2_deploy_time': 14.748, 'uav_3_smoke_2_explode_delay': 0.332,
        'uav_3_smoke_3_deploy_time': 57.396, 'uav_3_smoke_3_explode_delay': 5.967,
        # UAV4 烟幕弹
        'uav_4_smoke_1_deploy_time': 64.056, 'uav_4_smoke_1_explode_delay': 5.756,
        'uav_4_smoke_2_deploy_time': 63.184, 'uav_4_smoke_2_explode_delay': 4.504,
        'uav_4_smoke_3_deploy_time': 7.453, 'uav_4_smoke_3_explode_delay': 6.280,
        # UAV5 烟幕弹
        'uav_5_smoke_1_deploy_time': 4.385, 'uav_5_smoke_1_explode_delay': 1.933,
        'uav_5_smoke_2_deploy_time': 12.337, 'uav_5_smoke_2_explode_delay': 1.558,
        'uav_5_smoke_3_deploy_time': 32.446, 'uav_5_smoke_3_explode_delay': 5.838
    }
    
    # 创建可视化器并生成所有图表
    visualizer = Problem5Visualizer(best_solution=best_solution)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main() 