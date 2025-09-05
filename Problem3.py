"""
问题3：增强版粒子群优化算法求解单无人机三烟幕弹最优策略

改进特性：
1. 基于Problem2-PSO-Enhanced.py的框架
2. 支持两种遮蔽计算模式：独立遮蔽 vs 联合遮蔽
3. 优化8个决策变量：无人机参数 + 3个烟幕弹的时间参数
4. 自适应多种群并行搜索
5. 动态参数调整和重启机制
6. 自适应步长区间查找算法 (NEW)

核心优化：
- 使用Numba JIT编译加速核心几何计算
- 采用自适应步长算法优化时间区间查找
- LRU缓存减少重复计算
- 多烟幕弹协同遮蔽效应建模

目标：找到最优的无人机速度、飞行方向和3个烟幕弹的投放时间、引信延时，
使得有效遮蔽时长最大化。
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import copy

# 导入求解器
from solver import (
    calculate_single_uav_triple_smoke_masking,
    calculate_single_uav_triple_smoke_masking_multiple,
    TARGETS, MISSILES, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator


HAS_MULTIPLE_MASKING = False

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Particle:
    """粒子类"""
    position: np.ndarray
    velocity: np.ndarray
    fitness: float
    pbest_position: np.ndarray
    pbest_fitness: float
    stagnation_count: int = 0


# 全局函数，用于并行计算
def evaluate_particle_fitness_independent(particle_data):
    """评估粒子适应度的全局函数 - 独立遮蔽模式 (自适应步长)"""
    position, bounds_list = particle_data
    
    try:
        # 解码位置 - 8个决策变量
        params = {
            'v_FY1': position[0],           # 无人机速度
            'theta_FY1': position[1],       # 无人机方向
            'smoke_a_deploy_time': position[2],     # 烟幕弹A投放时间
            'smoke_a_explode_delay': position[3],   # 烟幕弹A引信延时
            'smoke_b_deploy_delay': position[4],    # 烟幕弹B相对A的投放延时
            'smoke_b_explode_delay': position[5],   # 烟幕弹B引信延时
            'smoke_c_deploy_delay': position[6],    # 烟幕弹C相对B的投放延时
            'smoke_c_explode_delay': position[7]    # 烟幕弹C引信延时
        }
        
        # 计算适应度 - 使用独立遮蔽模式 + 自适应步长算法
        duration = calculate_single_uav_triple_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_a_deploy_time=params['smoke_a_deploy_time'],
            smoke_a_explode_delay=params['smoke_a_explode_delay'],
            smoke_b_deploy_delay=params['smoke_b_deploy_delay'],
            smoke_b_explode_delay=params['smoke_b_explode_delay'],
            smoke_c_deploy_delay=params['smoke_c_deploy_delay'],
            smoke_c_explode_delay=params['smoke_c_explode_delay'],
            algorithm="adaptive"  # 使用自适应步长算法
        )
        
        return duration
        
    except Exception as e:
        print(f"独立遮蔽计算错误: {e}")
        return -1000.0


def evaluate_particle_fitness_multiple(particle_data):
    """评估粒子适应度的全局函数 - 联合遮蔽模式 (自适应步长)"""
    if not HAS_MULTIPLE_MASKING:
        return evaluate_particle_fitness_independent(particle_data)
    
    position, bounds_list = particle_data
    
    try:
        # 解码位置 - 8个决策变量
        params = {
            'v_FY1': position[0],           # 无人机速度
            'theta_FY1': position[1],       # 无人机方向
            'smoke_a_deploy_time': position[2],     # 烟幕弹A投放时间
            'smoke_a_explode_delay': position[3],   # 烟幕弹A引信延时
            'smoke_b_deploy_delay': position[4],    # 烟幕弹B相对A的投放延时
            'smoke_b_explode_delay': position[5],   # 烟幕弹B引信延时
            'smoke_c_deploy_delay': position[6],    # 烟幕弹C相对B的投放延时
            'smoke_c_explode_delay': position[7]    # 烟幕弹C引信延时
        }
        
        # 计算适应度 - 使用联合遮蔽模式
        duration = calculate_single_uav_triple_smoke_masking_multiple(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_a_deploy_time=params['smoke_a_deploy_time'],
            smoke_a_explode_delay=params['smoke_a_explode_delay'],
            smoke_b_deploy_delay=params['smoke_b_deploy_delay'],
            smoke_b_explode_delay=params['smoke_b_explode_delay'],
            smoke_c_deploy_delay=params['smoke_c_deploy_delay'],
            smoke_c_explode_delay=params['smoke_c_explode_delay']
        )
        
        return duration
        
    except Exception as e:
        print(f"联合遮蔽计算错误: {e}")
        return -1000.0


def calculate_bounds():
    """计算搜索空间边界"""
    print("正在计算搜索空间边界...")
    
    traj_calc = TrajectoryCalculator()
    missile_pos = MISSILES["M1"]["initial_pos"]
    missile_speed = MISSILES["M1"]["speed"]
    fake_target = TARGETS["fake_target"]
    
    distance = np.sqrt(
        (fake_target[0] - missile_pos[0])**2 + 
        (fake_target[1] - missile_pos[1])**2 + 
        (fake_target[2] - missile_pos[2])**2
    )
    t_max = distance / missile_speed
    
    print(f"导弹到达虚假目标时间: {t_max:.2f}s")
    
    # 8个决策变量的边界
    bounds = {
        'v_FY1': (70.0, 140.0),              # 无人机速度
        'theta_FY1': (0.0, 360.0),           # 无人机方向
        'smoke_a_deploy_time': (0.1, t_max - 5.0),    # 烟幕弹A投放时间
        'smoke_a_explode_delay': (0.1, 10.0),         # 烟幕弹A引信延时
        'smoke_b_deploy_delay': (0.1, 10.0),          # 烟幕弹B投放延时（相对A）
        'smoke_b_explode_delay': (0.1, 10.0),         # 烟幕弹B引信延时
        'smoke_c_deploy_delay': (0.1, 10.0),          # 烟幕弹C投放延时（相对B）
        'smoke_c_explode_delay': (0.1, 10.0)          # 烟幕弹C引信延时
    }
    
    return bounds


class EnhancedPSO_Problem3:
    """问题3增强版粒子群优化器"""
    
    def __init__(self,
                 n_particles: int = 40,
                 n_swarms: int = 4,
                 max_iterations: int = 200,
                 w_min: float = 0.1,
                 w_max: float = 0.9,
                 c1_initial: float = 2.5,
                 c2_initial: float = 0.5,
                 c1_final: float = 0.5,
                 c2_final: float = 2.5,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True,
                 restart_threshold: int = 25,
                 local_search_prob: float = 0.15,
                 masking_mode: str = "independent"):  # "independent" or "multiple"
        """
        初始化问题3增强版PSO优化器
        
        Args:
            masking_mode: 遮蔽计算模式
                - "independent": 独立遮蔽（任一烟幕弹满足条件即可）
                - "multiple": 联合遮蔽（考虑多烟幕弹协同效应）
        """
        self.n_particles = n_particles
        self.n_swarms = n_swarms
        self.max_iterations = max_iterations
        self.w_min = w_min
        self.w_max = w_max
        self.c1_initial = c1_initial
        self.c2_initial = c2_initial
        self.c1_final = c1_final
        self.c2_final = c2_final
        self.use_parallel = use_parallel
        self.restart_threshold = restart_threshold
        self.local_search_prob = local_search_prob
        self.masking_mode = masking_mode
        
        # 设置适应度评估函数
        if masking_mode == "multiple" and HAS_MULTIPLE_MASKING:
            self.fitness_function = evaluate_particle_fitness_multiple
            print("🔗 使用联合遮蔽模式")
        else:
            self.fitness_function = evaluate_particle_fitness_independent
            print("🔸 使用独立遮蔽模式")
        
        # 设置边界
        if bounds is None:
            self.bounds = calculate_bounds()
        else:
            self.bounds = bounds
            
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)  # 8个决策变量
        
        # 多种群
        self.swarms = []
        self.swarm_gbest = []
        self.swarm_gbest_fitness = []
        
        # 全局最优
        self.global_best_position = None
        self.global_best_fitness = -np.inf
        
        # 历史记录
        self.fitness_history = []
        self.diversity_history = []
        self.restart_count = 0
        
        # 并行计算设置
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), n_particles * n_swarms)
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
    
    def _create_particle(self) -> Particle:
        """创建一个随机粒子"""
        position = np.array([
            np.random.uniform(min_val, max_val) 
            for min_val, max_val in self.bounds_list
        ])
        
        # 初始速度较小
        velocity = np.array([
            np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
            for min_val, max_val in self.bounds_list
        ])
        
        return Particle(
            position=position,
            velocity=velocity,
            fitness=-np.inf,
            pbest_position=position.copy(),
            pbest_fitness=-np.inf
        )
    
    def _initialize_swarms(self):
        """初始化多个种群"""
        print(f"初始化 {self.n_swarms} 个种群，每个种群 {self.n_particles} 个粒子...")
        
        for swarm_idx in range(self.n_swarms):
            swarm = []
            for _ in range(self.n_particles):
                particle = self._create_particle()
                swarm.append(particle)
            
            self.swarms.append(swarm)
            self.swarm_gbest.append(None)
            self.swarm_gbest_fitness.append(-np.inf)
        
        # 评估初始适应度
        self._evaluate_all_particles()
        
        print(f"初始化完成，全局最优适应度: {self.global_best_fitness:.6f}")
    
    def _evaluate_all_particles(self):
        """评估所有粒子的适应度"""
        if self.use_parallel:
            # 并行计算
            all_particles = []
            particle_indices = []
            
            for swarm_idx, swarm in enumerate(self.swarms):
                for particle_idx, particle in enumerate(swarm):
                    all_particles.append((particle.position, self.bounds_list))
                    particle_indices.append((swarm_idx, particle_idx))
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {executor.submit(self.fitness_function, particle_data): i 
                          for i, particle_data in enumerate(all_particles)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    fitness_value = future.result()
                    swarm_idx, particle_idx = particle_indices[idx]
                    
                    particle = self.swarms[swarm_idx][particle_idx]
                    particle.fitness = fitness_value
                    
                    # 更新个体最优
                    if fitness_value > particle.pbest_fitness:
                        particle.pbest_fitness = fitness_value
                        particle.pbest_position = particle.position.copy()
                        particle.stagnation_count = 0
                    else:
                        particle.stagnation_count += 1
        else:
            # 串行计算
            for swarm_idx, swarm in enumerate(self.swarms):
                for particle_idx, particle in enumerate(swarm):
                    fitness_value = self.fitness_function((particle.position, self.bounds_list))
                    particle.fitness = fitness_value
                    
                    if fitness_value > particle.pbest_fitness:
                        particle.pbest_fitness = fitness_value
                        particle.pbest_position = particle.position.copy()
                        particle.stagnation_count = 0
                    else:
                        particle.stagnation_count += 1
        
        # 更新种群最优和全局最优
        self._update_global_best()
    
    def _update_global_best(self):
        """更新全局最优"""
        for swarm_idx, swarm in enumerate(self.swarms):
            # 更新种群最优
            best_particle = max(swarm, key=lambda p: p.fitness)
            if best_particle.fitness > self.swarm_gbest_fitness[swarm_idx]:
                self.swarm_gbest_fitness[swarm_idx] = best_particle.fitness
                self.swarm_gbest[swarm_idx] = best_particle.position.copy()
        
        # 更新全局最优
        best_swarm_idx = np.argmax(self.swarm_gbest_fitness)
        if self.swarm_gbest_fitness[best_swarm_idx] > self.global_best_fitness:
            self.global_best_fitness = self.swarm_gbest_fitness[best_swarm_idx]
            self.global_best_position = self.swarm_gbest[best_swarm_idx].copy()
    
    def _calculate_adaptive_weights(self, iteration: int) -> Tuple[float, float, float]:
        """计算自适应权重"""
        # 非线性惯性权重
        w = self.w_min + (self.w_max - self.w_min) * np.exp(-2 * iteration / self.max_iterations)
        
        # 时变学习因子
        c1 = (self.c1_final - self.c1_initial) * iteration / self.max_iterations + self.c1_initial
        c2 = (self.c2_final - self.c2_initial) * iteration / self.max_iterations + self.c2_initial
        
        return w, c1, c2
    
    def _local_search(self, particle: Particle):
        """局部搜索增强"""
        if np.random.random() > self.local_search_prob:
            return
        
        # 在当前最佳位置附近进行局部搜索
        search_radius = 0.1
        for _ in range(5):
            new_position = particle.pbest_position.copy()
            
            # 随机扰动
            for i in range(self.n_dims):
                min_val, max_val = self.bounds_list[i]
                range_val = max_val - min_val
                perturbation = np.random.normal(0, search_radius * range_val)
                new_position[i] += perturbation
                new_position[i] = np.clip(new_position[i], min_val, max_val)
            
            # 评估新位置
            fitness = self.fitness_function((new_position, self.bounds_list))
            
            if fitness > particle.pbest_fitness:
                particle.pbest_fitness = fitness
                particle.pbest_position = new_position.copy()
                particle.position = new_position.copy()
                particle.fitness = fitness
                break
    
    def _restart_stagnant_particles(self):
        """重启停滞的粒子"""
        for swarm in self.swarms:
            for particle in swarm:
                if particle.stagnation_count > self.restart_threshold:
                    # 重新初始化位置
                    particle.position = np.array([
                        np.random.uniform(min_val, max_val) 
                        for min_val, max_val in self.bounds_list
                    ])
                    
                    # 重置速度
                    particle.velocity = np.array([
                        np.random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
                        for min_val, max_val in self.bounds_list
                    ])
                    
                    particle.stagnation_count = 0
                    self.restart_count += 1
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        all_positions = []
        for swarm in self.swarms:
            for particle in swarm:
                all_positions.append(particle.position)
        
        if len(all_positions) < 2:
            return 0.0
        
        positions = np.array(all_positions)
        distances = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = np.linalg.norm(positions[i] - positions[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行增强版PSO优化"""
        print("="*60)
        print(f"开始问题3增强版粒子群优化算法 - {self.masking_mode.upper()}模式")
        print("="*60)
        
        # 初始化种群
        self._initialize_swarms()
        
        stagnation_count = 0
        best_fitness_unchanged = 0
        
        # 主优化循环
        for iteration in range(self.max_iterations):
            print(f"\n第 {iteration+1}/{self.max_iterations} 次迭代")
            
            # 计算自适应权重
            w, c1, c2 = self._calculate_adaptive_weights(iteration)
            
            # 更新所有粒子
            for swarm_idx, swarm in enumerate(self.swarms):
                swarm_gbest = self.swarm_gbest[swarm_idx]
                
                for particle in swarm:
                    # 生成随机数
                    r1 = np.random.random(self.n_dims)
                    r2 = np.random.random(self.n_dims)
                    r3 = np.random.random(self.n_dims)
                    
                    # 多样化的速度更新策略
                    if np.random.random() < 0.7:
                        # 标准PSO更新
                        particle.velocity = (
                            w * particle.velocity +
                            c1 * r1 * (particle.pbest_position - particle.position) +
                            c2 * r2 * (swarm_gbest - particle.position)
                        )
                    else:
                        # 全局最优引导
                        particle.velocity = (
                            w * particle.velocity +
                            c1 * r1 * (particle.pbest_position - particle.position) +
                            c2 * r2 * (self.global_best_position - particle.position) +
                            0.1 * r3 * (np.random.random(self.n_dims) - 0.5)  # 随机扰动
                        )
                    
                    # 速度限制
                    for i in range(self.n_dims):
                        min_val, max_val = self.bounds_list[i]
                        v_max = 0.2 * (max_val - min_val)
                        particle.velocity[i] = np.clip(particle.velocity[i], -v_max, v_max)
                    
                    # 更新位置
                    particle.position = particle.position + particle.velocity
                    
                    # 边界处理
                    for i in range(self.n_dims):
                        min_val, max_val = self.bounds_list[i]
                        particle.position[i] = np.clip(particle.position[i], min_val, max_val)
                    
                    # 局部搜索
                    self._local_search(particle)
            
            # 评估所有粒子
            self._evaluate_all_particles()
            
            # 计算多样性
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # 记录历史
            self.fitness_history.append(self.global_best_fitness)
            
            # 重启停滞粒子
            self._restart_stagnant_particles()
            
            # 输出信息
            avg_fitness = np.mean([p.fitness for swarm in self.swarms for p in swarm])
            print(f"  惯性权重: {w:.3f}, c1: {c1:.3f}, c2: {c2:.3f}")
            print(f"  全局最优: {self.global_best_fitness:.6f}")
            print(f"  平均适应度: {avg_fitness:.6f}")
            print(f"  种群多样性: {diversity:.6f}")
            print(f"  重启次数: {self.restart_count}")
            
            # 检查是否达到很好的结果
            if self.global_best_fitness >= 10.0:  # 三烟幕弹的理论上限可能更高
                print(f"  🎯 发现优秀解！")
            
            # 收敛检查
            if iteration > 20:
                recent_improvement = max(self.fitness_history[-10:]) - min(self.fitness_history[-10:])
                if recent_improvement < 1e-6:
                    stagnation_count += 1
                else:
                    stagnation_count = 0
                
                if stagnation_count > 15:
                    print(f"  算法收敛，提前结束于第 {iteration+1} 次迭代")
                    break
        
        return self.global_best_position, self.global_best_fitness
    
    def plot_convergence(self):
        """绘制详细的收敛分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 适应度收敛曲线
        axes[0, 0].plot(self.fitness_history, 'b-', linewidth=2, label='全局最优')
        axes[0, 0].set_title(f'问题3增强版PSO收敛曲线 ({self.masking_mode.upper()}模式)')
        axes[0, 0].set_xlabel('迭代次数')
        axes[0, 0].set_ylabel('适应度（有效遮蔽时长）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 多样性变化
        axes[0, 1].plot(self.diversity_history, 'g-', linewidth=2)
        axes[0, 1].set_title('种群多样性变化')
        axes[0, 1].set_xlabel('迭代次数')
        axes[0, 1].set_ylabel('多样性')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 收敛速度
        if len(self.fitness_history) > 1:
            improvements = np.diff(self.fitness_history)
            axes[1, 0].plot(improvements, 'orange', linewidth=2)
            axes[1, 0].set_title('适应度改进速度')
            axes[1, 0].set_xlabel('迭代次数')
            axes[1, 0].set_ylabel('适应度提升')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 参数分布（最后一代）
        all_positions = []
        for swarm in self.swarms:
            for particle in swarm:
                all_positions.append(particle.position)
        
        if all_positions:
            positions = np.array(all_positions)
            axes[1, 1].hist(positions[:, 0], bins=20, alpha=0.7, label='速度分布')
            axes[1, 1].set_title('参数分布（无人机速度）')
            axes[1, 1].set_xlabel('无人机速度 (m/s)')
            axes[1, 1].set_ylabel('粒子数量')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_problem3_results(best_position: np.ndarray, best_fitness: float, 
                           bounds: Dict[str, Tuple[float, float]], masking_mode: str):
    """分析问题3结果"""
    print("="*60)
    print(f"问题3优化结果分析 ({masking_mode.upper()}模式)")
    print("="*60)
    
    # 解码最优解
    keys = list(bounds.keys())
    best_params = {keys[i]: best_position[i] for i in range(len(keys))}
    
    print(f"\n最优策略参数：")
    print(f"  无人机速度 (v_FY1): {best_params['v_FY1']:.2f} m/s")
    print(f"  无人机方向 (θ_FY1): {best_params['theta_FY1']:.2f}°")
    print(f"  烟幕弹A投放时间: {best_params['smoke_a_deploy_time']:.3f} s")
    print(f"  烟幕弹A引信延时: {best_params['smoke_a_explode_delay']:.3f} s")
    print(f"  烟幕弹A起爆时间: {best_params['smoke_a_deploy_time'] + best_params['smoke_a_explode_delay']:.3f} s")
    
    # 计算烟幕弹B的绝对时间
    smoke_b_deploy_time = best_params['smoke_a_deploy_time'] + best_params['smoke_b_deploy_delay']
    smoke_b_explode_time = smoke_b_deploy_time + best_params['smoke_b_explode_delay']
    print(f"  烟幕弹B投放时间: {smoke_b_deploy_time:.3f} s (延时: {best_params['smoke_b_deploy_delay']:.3f} s)")
    print(f"  烟幕弹B引信延时: {best_params['smoke_b_explode_delay']:.3f} s")
    print(f"  烟幕弹B起爆时间: {smoke_b_explode_time:.3f} s")
    
    # 计算烟幕弹C的绝对时间
    smoke_c_deploy_time = smoke_b_deploy_time + best_params['smoke_c_deploy_delay']
    smoke_c_explode_time = smoke_c_deploy_time + best_params['smoke_c_explode_delay']
    print(f"  烟幕弹C投放时间: {smoke_c_deploy_time:.3f} s (延时: {best_params['smoke_c_deploy_delay']:.3f} s)")
    print(f"  烟幕弹C引信延时: {best_params['smoke_c_explode_delay']:.3f} s")
    print(f"  烟幕弹C起爆时间: {smoke_c_explode_time:.3f} s")
    
    print(f"\n最大有效遮蔽时长: {best_fitness:.6f} 秒")
    
    # 验证结果
    print(f"\n验证计算...")
    if masking_mode == "multiple" and HAS_MULTIPLE_MASKING:
        verification_result = calculate_single_uav_triple_smoke_masking_multiple(
            uav_direction=best_params['theta_FY1'],
            uav_speed=best_params['v_FY1'],
            smoke_a_deploy_time=best_params['smoke_a_deploy_time'],
            smoke_a_explode_delay=best_params['smoke_a_explode_delay'],
            smoke_b_deploy_delay=best_params['smoke_b_deploy_delay'],
            smoke_b_explode_delay=best_params['smoke_b_explode_delay'],
            smoke_c_deploy_delay=best_params['smoke_c_deploy_delay'],
            smoke_c_explode_delay=best_params['smoke_c_explode_delay']
        )
    else:
        verification_result = calculate_single_uav_triple_smoke_masking(
            uav_direction=best_params['theta_FY1'],
            uav_speed=best_params['v_FY1'],
            smoke_a_deploy_time=best_params['smoke_a_deploy_time'],
            smoke_a_explode_delay=best_params['smoke_a_explode_delay'],
            smoke_b_deploy_delay=best_params['smoke_b_deploy_delay'],
            smoke_b_explode_delay=best_params['smoke_b_explode_delay'],
            smoke_c_deploy_delay=best_params['smoke_c_deploy_delay'],
            smoke_c_explode_delay=best_params['smoke_c_explode_delay'],
            algorithm="adaptive"  # 使用自适应步长算法验证
        )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    return best_params


def compare_masking_modes():
    """比较两种遮蔽模式的性能"""
    if not HAS_MULTIPLE_MASKING:
        print("⚠️  无法进行模式比较，缺少联合遮蔽函数")
        return
    
    print("="*60)
    print("遮蔽模式比较分析")
    print("="*60)
    
    # 共同的PSO参数
    pso_params = {
        'n_particles': 30,
        'n_swarms': 3,
        'max_iterations': 100,  # 减少迭代次数用于比较
        'use_parallel': True
    }
    
    results = {}
    
    # 测试独立遮蔽模式
    print("\n🔸 测试独立遮蔽模式...")
    optimizer_independent = EnhancedPSO_Problem3(masking_mode="independent", **pso_params)
    start_time = time.time()
    best_pos_ind, best_fit_ind = optimizer_independent.optimize()
    time_ind = time.time() - start_time
    results['independent'] = {
        'fitness': best_fit_ind,
        'time': time_ind,
        'params': {list(optimizer_independent.bounds.keys())[i]: best_pos_ind[i] 
                  for i in range(len(best_pos_ind))}
    }
    
    # 测试联合遮蔽模式
    print("\n🔗 测试联合遮蔽模式...")
    optimizer_multiple = EnhancedPSO_Problem3(masking_mode="multiple", **pso_params)
    start_time = time.time()
    best_pos_mul, best_fit_mul = optimizer_multiple.optimize()
    time_mul = time.time() - start_time
    results['multiple'] = {
        'fitness': best_fit_mul,
        'time': time_mul,
        'params': {list(optimizer_multiple.bounds.keys())[i]: best_pos_mul[i] 
                  for i in range(len(best_pos_mul))}
    }
    
    # 比较结果
    print("\n" + "="*60)
    print("比较结果")
    print("="*60)
    
    print(f"独立遮蔽模式:")
    print(f"  最优遮蔽时长: {results['independent']['fitness']:.6f} 秒")
    print(f"  优化耗时: {results['independent']['time']:.2f} 秒")
    
    print(f"\n联合遮蔽模式:")
    print(f"  最优遮蔽时长: {results['multiple']['fitness']:.6f} 秒")
    print(f"  优化耗时: {results['multiple']['time']:.2f} 秒")
    
    # 性能比较
    fitness_improvement = results['multiple']['fitness'] - results['independent']['fitness']
    time_ratio = results['multiple']['time'] / results['independent']['time']
    
    print(f"\n性能分析:")
    print(f"  遮蔽时长提升: {fitness_improvement:+.6f} 秒 ({fitness_improvement/results['independent']['fitness']*100:+.2f}%)")
    print(f"  计算时间比: {time_ratio:.2f}x")
    
    if fitness_improvement > 0:
        print(f"  ✅ 联合遮蔽模式表现更优")
    else:
        print(f"  ⚠️  独立遮蔽模式表现更优")
    
    return results


def main():
    """主函数"""
    print("问题3：单无人机三烟幕弹最优策略求解")
    
    # 设置PSO参数
    pso_params = {
        'n_particles': 40,           # 每个种群粒子数
        'n_swarms': 4,               # 种群数量
        'max_iterations': 250,       # 最大迭代次数（增加以应对更复杂的搜索空间）
        'w_min': 0.1,               # 最小惯性权重
        'w_max': 0.9,               # 最大惯性权重
        'c1_initial': 2.5,          # 初始个体学习因子
        'c2_initial': 0.5,          # 初始全局学习因子
        'c1_final': 0.5,            # 终止个体学习因子
        'c2_final': 2.5,            # 终止全局学习因子
        'use_parallel': True,        # 使用并行计算
        'restart_threshold': 30,     # 重启阈值
        'local_search_prob': 0.15    # 局部搜索概率
    }
    
    print(f"\n问题3增强版PSO算法参数：")
    for key, value in pso_params.items():
        print(f"  {key}: {value}")
    
    # 选择遮蔽模式
    masking_mode = "multiple" if HAS_MULTIPLE_MASKING else "independent"
    
    # 创建优化器
    optimizer = EnhancedPSO_Problem3(masking_mode=masking_mode, **pso_params)
    
    print(f"\n搜索空间边界（8个决策变量）：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    best_position, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    
    # 分析结果
    best_params = analyze_problem3_results(best_position, best_fitness, 
                                         optimizer.bounds, masking_mode)
    
    # 绘制收敛曲线
    optimizer.plot_convergence()
    
    # 保存结果
    results = {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'pso_params': pso_params,
        'bounds': optimizer.bounds,
        'fitness_history': optimizer.fitness_history,
        'diversity_history': optimizer.diversity_history,
        'restart_count': optimizer.restart_count,
        'masking_mode': masking_mode
    }
    
    print(f"\n问题3优化结果已保存")
    
    # 如果支持，进行模式比较
    if HAS_MULTIPLE_MASKING:
        print(f"\n是否进行遮蔽模式比较？(y/n): ", end="")
        try:
            choice = input().lower()
            if choice == 'y':
                compare_masking_modes()
        except:
            pass
    
    return results


if __name__ == "__main__":
    results = main()