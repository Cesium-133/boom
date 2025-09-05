"""
问题2可视化脚本 - PSO增强版优化结果可视化

本脚本生成以下图表：
1. 最优策略下的三维轨迹图
2. 解空间分析图 (灵敏度分析)

功能：
- 基于Problem2-PSO-Enhanced.py的最优参数生成可视化
- 分析各参数对遮蔽时长的影响
- 保存高质量图像文件
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
from datetime import datetime
import sys

# 导入求解器模块
sys.path.append('..')
from solver import (
    calculate_single_uav_single_smoke_masking,
    CONSTANTS, TARGETS, MISSILES, UAVS, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator
from solver.geometry import get_top_plane_points, get_under_points

# 导入PSO优化器
sys.path.append('..')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("problem2_pso", "../Problem2-PSO-Enhanced.py")
    problem2_pso = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(problem2_pso)
    calculate_bounds = problem2_pso.calculate_bounds
except:
    # 如果无法导入，定义一个简单的边界计算函数
    def calculate_bounds():
        return {
            'v_FY1': (70.0, 140.0),
            'theta_FY1': (0.0, 360.0),
            't_deploy': (0.1, 65.0),
            't_fuse': (0.1, 65.0)
        }

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


class Problem2Visualizer:
    """问题2可视化器"""
    
    def __init__(self, best_params=None):
        """
        初始化可视化器
        
        Args:
            best_params: 最优参数字典，包含 v_FY1, theta_FY1, t_deploy, t_fuse
        """
        # 如果没有提供最优参数，使用默认的较优参数
        if best_params is None:
            self.best_params = {
                'v_FY1': 120.0,      # 无人机速度
                'theta_FY1': 90.0,   # 无人机方向
                't_deploy': 1.5,     # 投放时间
                't_fuse': 3.6        # 引信延时
            }
        else:
            self.best_params = best_params
            
        self.max_time = 50.0
        
        # 创建轨迹计算器
        self.traj = TrajectoryCalculator()
        self.missile_traj = self.traj.create_missile_trajectory("M1")
        self.uav_traj = self.traj.create_uav_trajectory("FY1", 
                                                        direction_degrees=self.best_params['theta_FY1'], 
                                                        speed=self.best_params['v_FY1'])
        self.smoke_traj = self.traj.create_smoke_trajectory(self.uav_traj, 
                                                           self.best_params['t_deploy'], 
                                                           self.best_params['t_fuse'])
        
        # 计算关键时间点
        self.explode_time = self.best_params['t_deploy'] + self.best_params['t_fuse']
        
        # 预计算轨迹数据
        self.times = np.linspace(0, self.max_time, 1000)
        self.missile_pts = np.array([self.missile_traj(t) for t in self.times])
        self.uav_pts = np.array([self.uav_traj(t) for t in self.times])
        self.smoke_pts = np.array([self.smoke_traj(t) for t in self.times])
        
        # 计算最优遮蔽时长
        self.optimal_duration = calculate_single_uav_single_smoke_masking(
            uav_direction=self.best_params['theta_FY1'],
            uav_speed=self.best_params['v_FY1'],
            smoke_deploy_time=self.best_params['t_deploy'],
            smoke_explode_delay=self.best_params['t_fuse']
        )
        
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
    
    def plot_optimal_3d_trajectory(self):
        """最优策略下的三维轨迹图"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制完整轨迹
        ax.plot(self.missile_pts[:, 0], self.missile_pts[:, 1], 
                self.missile_pts[:, 2], 'b-', linewidth=2, label='导弹M1轨迹')
        ax.plot(self.uav_pts[:, 0], self.uav_pts[:, 1], 
                self.uav_pts[:, 2], 'r-', linewidth=2, label='无人机FY1轨迹')
        
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
        deploy_pos = self.uav_traj(self.best_params['t_deploy'])
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
        ax.set_title('问题2 - 最优策略下的三维轨迹图', fontsize=16, weight='bold')
        
        # 设置视角
        ax.view_init(elev=25, azim=60)
        
        # 设置坐标范围 - 调整比例突出关键区域
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
        
        # 调整坐标轴刻度间距
        ax.set_xticks(np.arange(6000, 20000, 2000))
        ax.set_yticks(np.arange(-500, 1000, 250))
        ax.set_zticks(np.arange(0, 2500, 500))
        
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))
        
        # 添加最优参数信息 - 调整位置避免与图例重合
        param_text = f"最优参数:\n"
        param_text += f"无人机速度: {self.best_params['v_FY1']:.1f} m/s\n"
        param_text += f"飞行方向: {self.best_params['theta_FY1']:.1f}°\n"
        param_text += f"投放时间: {self.best_params['t_deploy']:.2f} s\n"
        param_text += f"引信延时: {self.best_params['t_fuse']:.2f} s\n"
        param_text += f"遮蔽时长: {self.optimal_duration:.2f} s"
        
        ax.text2D(0.02, 0.75, param_text, transform=ax.transAxes, 
                  fontsize=10, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题2_最优策略三维轨迹图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"最优策略三维轨迹图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def plot_sensitivity_analysis(self):
        """解空间分析图 (灵敏度分析)"""
        # 获取参数边界
        bounds = calculate_bounds()
        
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('问题2 - 解空间分析图 (灵敏度分析)', fontsize=16, weight='bold')
        
        # 参数定义
        param_configs = [
            {'key': 'v_FY1', 'name': '无人机速度', 'unit': 'm/s', 'ax': axes[0, 0]},
            {'key': 'theta_FY1', 'name': '飞行方向', 'unit': '°', 'ax': axes[0, 1]},
            {'key': 't_deploy', 'name': '投放时间', 'unit': 's', 'ax': axes[1, 0]},
            {'key': 't_fuse', 'name': '引信延时', 'unit': 's', 'ax': axes[1, 1]}
        ]
        
        for config in param_configs:
            param_key = config['key']
            param_name = config['name']
            param_unit = config['unit']
            ax = config['ax']
            
            print(f"正在分析 {param_name} 的灵敏度...")
            
            # 生成参数变化范围
            param_min, param_max = bounds[param_key]
            if param_key == 'theta_FY1':
                # 对于角度，使用更细的采样
                param_values = np.linspace(param_min, param_max, 72)  # 每5度一个点
            else:
                param_values = np.linspace(param_min, param_max, 50)
            
            durations = []
            
            # 计算每个参数值对应的遮蔽时长
            for param_value in param_values:
                # 创建临时参数字典，只改变当前分析的参数
                temp_params = self.best_params.copy()
                temp_params[param_key] = param_value
                
                try:
                    duration = calculate_single_uav_single_smoke_masking(
                        uav_direction=temp_params['theta_FY1'],
                        uav_speed=temp_params['v_FY1'],
                        smoke_deploy_time=temp_params['t_deploy'],
                        smoke_explode_delay=temp_params['t_fuse']
                    )
                    durations.append(duration)
                except:
                    durations.append(0.0)  # 如果计算失败，设为0
            
            # 绘制曲线
            ax.plot(param_values, durations, 'b-', linewidth=2, label='遮蔽时长')
            
            # 标记最优点 - 计算最优参数对应的实际遮蔽时长
            optimal_value = self.best_params[param_key]
            
            # 计算最优参数在当前变化中对应的遮蔽时长
            temp_params = self.best_params.copy()
            temp_params[param_key] = optimal_value
            try:
                optimal_duration_for_this_param = calculate_single_uav_single_smoke_masking(
                    uav_direction=temp_params['theta_FY1'],
                    uav_speed=temp_params['v_FY1'],
                    smoke_deploy_time=temp_params['t_deploy'],
                    smoke_explode_delay=temp_params['t_fuse']
                )
            except:
                optimal_duration_for_this_param = self.optimal_duration
            
            ax.axvline(x=optimal_value, color='red', linestyle='--', alpha=0.7, label='最优值')
            ax.scatter([optimal_value], [optimal_duration_for_this_param], 
                      color='red', s=100, marker='o', zorder=5)
            
            # 设置坐标轴
            ax.set_xlabel(f'{param_name} ({param_unit})', fontsize=12)
            ax.set_ylabel('遮蔽时长 (s)', fontsize=12)
            ax.set_title(f'{param_name}灵敏度分析', fontsize=14, weight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # 添加统计信息
            max_duration = max(durations)
            min_duration = min(durations)
            sensitivity = max_duration - min_duration
            
            info_text = f"最大值: {max_duration:.2f}s\n"
            info_text += f"最小值: {min_duration:.2f}s\n"
            info_text += f"灵敏度: {sensitivity:.2f}s\n"
            info_text += f"最优值: {optimal_value:.2f}"
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图像
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"问题2_解空间分析图_{timestamp}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"解空间分析图已保存: {filepath}")
        
        plt.show()
        return fig
    
    def generate_all_plots(self):
        """生成所有图表"""
        print("=" * 60)
        print("问题2可视化 - 生成优化结果图表")
        print("=" * 60)
        
        print(f"最优参数：")
        print(f"  无人机速度: {self.best_params['v_FY1']:.2f} m/s")
        print(f"  飞行方向: {self.best_params['theta_FY1']:.2f}°")
        print(f"  投放时间: {self.best_params['t_deploy']:.2f} s")
        print(f"  引信延时: {self.best_params['t_fuse']:.2f} s")
        print(f"  最优遮蔽时长: {self.optimal_duration:.2f} s")
        print()
        
        # 生成图表
        print("正在生成最优策略下的三维轨迹图...")
        self.plot_optimal_3d_trajectory()
        
        print("正在生成解空间分析图...")
        self.plot_sensitivity_analysis()
        
        print(f"\n所有图表已保存到 {self.output_dir} 文件夹")
        print("生成完成！")


def main():
    """主函数"""
    # 这里可以设置从PSO优化得到的最优参数
    # 如果没有运行PSO，使用默认的较好参数
    best_params = {
        'v_FY1': 140.0,      # 无人机速度
        'theta_FY1': 5.15,   # 无人机方向 (朝向虚假目标)
        't_deploy': 0.769,     # 投放时间
        't_fuse': 0.156        # 引信延时
    }
    
    # 如果想要使用真实的PSO优化结果，可以先运行Problem2-PSO-Enhanced.py
    # 然后在这里加载结果
    
    # 创建可视化器并生成所有图表
    visualizer = Problem2Visualizer(best_params=best_params)
    visualizer.generate_all_plots()


if __name__ == "__main__":
    main() 