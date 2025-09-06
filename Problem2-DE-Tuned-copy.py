"""
问题2：差分进化算法求解最优无人机策略 - 精细调参版本

专门针对接近最优解(4.8)时的精细调参策略
当前收敛到4.588，目标提升到4.8+
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import copy

# 导入求解器
from solver import calculate_single_uav_single_smoke_masking, TARGETS, MISSILES
from solver.trajectory import TrajectoryCalculator
from solver.core import MaskingCalculator, find_t_intervals, find_t_intervals_adaptive, find_t_intervals_smart

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Individual:
    """个体类"""
    position: np.ndarray
    fitness: float
    generation: int = 0


def evaluate_individual_fitness(individual_data):
    """评估个体适应度的全局函数"""
    position, bounds_list = individual_data
    
    try:
        params = {
            'v_FY1': position[0],
            'theta_FY1': position[1],
            't_deploy': position[2],
            't_fuse': position[3]
        }
        
        duration = calculate_single_uav_single_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_deploy_time=params['t_deploy'],
            smoke_explode_delay=params['t_fuse'],
            algorithm="adaptive"
        )
        
        return duration
        
    except Exception as e:
        print(f"计算错误: {e}")
        return -1000.0


def compute_intervals_for_params(uav_direction: float,
                                 uav_speed: float,
                                 smoke_deploy_time: float,
                                 smoke_explode_delay: float,
                                 algorithm: str = "adaptive") -> Tuple[List[Tuple[float, float]], float]:
    """根据参数计算遮蔽区间与总时长，用于中断时展示。"""
    calc = MaskingCalculator()

    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    uav_traj = calc.traj_calc.create_uav_trajectory("FY1", direction_degrees=uav_direction, speed=uav_speed)
    smoke_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_deploy_time, smoke_explode_delay)

    predicate = calc._create_masking_predicate(missile_traj, smoke_traj)
    explode_time = smoke_deploy_time + smoke_explode_delay
    _, end_time = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)

    if algorithm == "fixed":
        intervals = find_t_intervals(predicate, calc.threshold, explode_time, end_time, calc.time_step)
    elif algorithm == "adaptive":
        intervals = find_t_intervals_adaptive(
            predicate,
            calc.threshold,
            explode_time,
            end_time,
            initial_step=calc.time_step * 10,
            min_step=calc.time_step / 2,
            max_step=calc.time_step * 50
        )
    elif algorithm == "smart":
        intervals = find_t_intervals_smart(
            predicate,
            calc.threshold,
            explode_time,
            end_time,
            initial_step=calc.time_step * 5,
            aggressive_speedup=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    duration = float(sum(b - a for a, b in intervals))
    return intervals, duration


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
    
    bounds = {
        'v_FY1': (70.0, 140.0),
        'theta_FY1': (0.0, 360.0),
        't_deploy': (0.001, 60),
        't_fuse': (0.001, 60)
    }
    
    return bounds


class TunedDifferentialEvolution:
    """精细调参版差分进化算法"""
    
    def __init__(self,
                 population_size: int = 80,           # 增大种群
                 max_generations: int = 800,          # 增加代数
                 F_min: float = 0.3,                 # 降低最小F，增强精细搜索
                 F_max: float = 1.2,                 # 提高最大F，增强全局搜索
                 CR_min: float = 0.05,               # 降低最小CR
                 CR_max: float = 0.95,               # 提高最大CR
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True,
                 restart_threshold: int = 50,        # 添加重启机制
                 local_search_prob: float = 0.1,     # 添加局部搜索
                 multi_population: bool = True):      # 多子种群
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.use_parallel = use_parallel
        self.restart_threshold = restart_threshold
        self.local_search_prob = local_search_prob
        self.multi_population = multi_population
        
        # 设置边界
        if bounds is None:
            self.bounds = calculate_bounds()
        else:
            self.bounds = bounds
            
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)
        
        # 多种变异策略（增加更多策略）
        self.mutation_strategies = [
            'DE/rand/1',
            'DE/best/1', 
            'DE/current-to-best/1',
            'DE/rand/2',
            'DE/best/2',
            'DE/rand-to-best/1',      # 新增
            'DE/current-to-rand/1'    # 新增
        ]
        
        # 多子种群设置
        if multi_population:
            self.n_subpopulations = 4
            self.subpop_size = population_size // self.n_subpopulations
            self.migration_interval = 20  # 每20代进行一次迁移
        else:
            self.n_subpopulations = 1
            self.subpop_size = population_size
        
        # 初始化
        self.population = []
        self.subpopulations = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # 历史记录
        self.fitness_history = []
        self.diversity_history = []
        self.stagnation_count = 0
        self.restart_count = 0
        
        # 并行计算设置
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), population_size)
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
    
    def _initialize_population(self):
        """初始化种群（支持多子种群）"""
        print(f"初始化种群，大小: {self.population_size}")
        
        if self.multi_population:
            # 多子种群初始化
            self.subpopulations = []
            for sub_idx in range(self.n_subpopulations):
                subpop = []
                for _ in range(self.subpop_size):
                    position = np.array([
                        np.random.uniform(min_val, max_val) 
                        for min_val, max_val in self.bounds_list
                    ])
                    individual = Individual(position=position, fitness=-np.inf)
                    subpop.append(individual)
                self.subpopulations.append(subpop)
                
            # 合并所有子种群
            self.population = []
            for subpop in self.subpopulations:
                self.population.extend(subpop)
        else:
            # 单种群初始化
            self.population = []
            for _ in range(self.population_size):
                position = np.array([
                    np.random.uniform(min_val, max_val) 
                    for min_val, max_val in self.bounds_list
                ])
                individual = Individual(position=position, fitness=-np.inf)
                self.population.append(individual)
        
        # 评估初始种群
        self._evaluate_population()
        self._update_best()
        
        print(f"初始化完成，最佳适应度: {self.best_fitness:.6f}")
    
    def _evaluate_population(self):
        """评估种群适应度"""
        if self.use_parallel:
            individual_data = [(ind.position, self.bounds_list) for ind in self.population]
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {executor.submit(evaluate_individual_fitness, data): i 
                          for i, data in enumerate(individual_data)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    fitness_value = future.result()
                    self.population[idx].fitness = fitness_value
        else:
            for individual in self.population:
                fitness_value = evaluate_individual_fitness((individual.position, self.bounds_list))
                individual.fitness = fitness_value
    
    def _update_best(self):
        """更新最佳个体"""
        prev_best = self.best_fitness
        for individual in self.population:
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = copy.deepcopy(individual)
        
        # 更新停滞计数
        if self.best_fitness <= prev_best + 1e-8:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0
    
    def _adaptive_parameters(self, generation: int) -> Tuple[float, float]:
        """增强的自适应参数调整"""
        progress = generation / self.max_generations
        
        # 更精细的F调整策略
        if self.best_fitness < 4.0:
            # 初期：大F值，增强全局搜索
            F = self.F_max - (self.F_max - 0.8) * progress
        elif self.best_fitness < 4.5:
            # 中期：中等F值，平衡探索和开发
            F = 0.8 - (0.8 - 0.5) * progress
        else:
            # 后期：小F值，精细搜索
            F = 0.5 - (0.5 - self.F_min) * progress
        
        # 基于停滞情况调整F
        if self.stagnation_count > 20:
            F = min(self.F_max, F * 1.5)  # 增强探索
        
        # CR调整：基于种群多样性和适应度
        diversity = self._calculate_diversity()
        if diversity < 0.1 and self.best_fitness > 4.5:
            # 多样性低且接近最优解时，增加交叉概率
            CR = self.CR_max
        elif diversity > 0.5:
            # 多样性高时，适中交叉概率
            CR = (self.CR_min + self.CR_max) / 2
        else:
            # 正常情况下的自适应调整
            CR = self.CR_min + (self.CR_max - self.CR_min) * (1 - progress)
        
        return F, CR
    
    def _enhanced_mutation(self, target_idx: int, F: float, generation: int) -> np.ndarray:
        """增强的变异操作"""
        # 根据当前适应度选择变异策略
        if self.best_fitness < 4.0:
            # 初期：偏向探索性策略
            strategies = ['DE/rand/1', 'DE/rand/2', 'DE/current-to-rand/1']
        elif self.best_fitness < 4.5:
            # 中期：平衡策略
            strategies = ['DE/best/1', 'DE/current-to-best/1', 'DE/rand-to-best/1']
        else:
            # 后期：偏向开发性策略
            strategies = ['DE/best/1', 'DE/best/2', 'DE/current-to-best/1']
        
        strategy = np.random.choice(strategies)
        
        population_indices = list(range(len(self.population)))
        population_indices.remove(target_idx)
        
        if strategy == 'DE/rand/1':
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position))
        
        elif strategy == 'DE/best/1':
            r1, r2 = np.random.choice(population_indices, 2, replace=False)
            mutant = (self.best_individual.position + 
                     F * (self.population[r1].position - self.population[r2].position))
        
        elif strategy == 'DE/current-to-best/1':
            r1, r2 = np.random.choice(population_indices, 2, replace=False)
            mutant = (self.population[target_idx].position + 
                     F * (self.best_individual.position - self.population[target_idx].position) +
                     F * (self.population[r1].position - self.population[r2].position))
        
        elif strategy == 'DE/rand/2':
            r1, r2, r3, r4, r5 = np.random.choice(population_indices, 5, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position) +
                     F * (self.population[r4].position - self.population[r5].position))
        
        elif strategy == 'DE/best/2':
            r1, r2, r3, r4 = np.random.choice(population_indices, 4, replace=False)
            mutant = (self.best_individual.position + 
                     F * (self.population[r1].position - self.population[r2].position) +
                     F * (self.population[r3].position - self.population[r4].position))
        
        elif strategy == 'DE/rand-to-best/1':
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.best_individual.position - self.population[r1].position) +
                     F * (self.population[r2].position - self.population[r3].position))
        
        elif strategy == 'DE/current-to-rand/1':
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[target_idx].position + 
                     F * (self.population[r1].position - self.population[target_idx].position) +
                     F * (self.population[r2].position - self.population[r3].position))
        
        # 边界处理
        for i in range(self.n_dims):
            min_val, max_val = self.bounds_list[i]
            mutant[i] = np.clip(mutant[i], min_val, max_val)
        
        return mutant
    
    def _local_search(self, individual: Individual) -> Individual:
        """局部搜索增强"""
        if np.random.random() > self.local_search_prob:
            return individual
        
        best_local = copy.deepcopy(individual)
        search_radius = 0.05  # 小范围搜索
        
        for _ in range(5):  # 尝试5次局部搜索
            new_position = individual.position.copy()
            
            # 在当前位置附近随机扰动
            for i in range(self.n_dims):
                min_val, max_val = self.bounds_list[i]
                range_val = max_val - min_val
                perturbation = np.random.normal(0, search_radius * range_val)
                new_position[i] = np.clip(new_position[i] + perturbation, min_val, max_val)
            
            # 评估新位置
            fitness = evaluate_individual_fitness((new_position, self.bounds_list))
            
            if fitness > best_local.fitness:
                best_local.position = new_position
                best_local.fitness = fitness
        
        return best_local
    
    def _migration(self, generation: int):
        """子种群间迁移"""
        if not self.multi_population or generation % self.migration_interval != 0:
            return
        
        # 每个子种群选择最好的个体进行迁移
        migrants = []
        for subpop in self.subpopulations:
            best_in_subpop = max(subpop, key=lambda x: x.fitness)
            migrants.append(copy.deepcopy(best_in_subpop))
        
        # 环形迁移
        for i in range(self.n_subpopulations):
            target_subpop = (i + 1) % self.n_subpopulations
            # 用迁移个体替换目标子种群中最差的个体
            worst_idx = min(range(len(self.subpopulations[target_subpop])), 
                           key=lambda x: self.subpopulations[target_subpop][x].fitness)
            self.subpopulations[target_subpop][worst_idx] = migrants[i]
    
    def _restart_mechanism(self):
        """重启机制"""
        if self.stagnation_count < self.restart_threshold:
            return
        
        print(f"    触发重启机制 (停滞{self.stagnation_count}代)")
        
        # 保留最好的20%个体
        n_keep = int(0.2 * self.population_size)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elite = self.population[:n_keep]
        
        # 重新初始化其余个体
        for i in range(n_keep, self.population_size):
            position = np.array([
                np.random.uniform(min_val, max_val) 
                for min_val, max_val in self.bounds_list
            ])
            self.population[i] = Individual(position=position, fitness=-np.inf)
        
        # 重新评估
        self._evaluate_population()
        self.restart_count += 1
        self.stagnation_count = 0
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
        
        positions = np.array([ind.position for ind in self.population])
        distances = []
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                normalized_dist = 0
                for k in range(self.n_dims):
                    min_val, max_val = self.bounds_list[k]
                    normalized_dist += ((positions[i][k] - positions[j][k]) / (max_val - min_val)) ** 2
                distances.append(np.sqrt(normalized_dist))
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行精细调参的差分进化优化"""
        print("="*60)
        print("开始精细调参差分进化算法优化")
        print("="*60)
        
        # 初始化种群
        self._initialize_population()
        
        # 主优化循环
        try:
            for generation in range(self.max_generations):
                print(f"\n第 {generation+1}/{self.max_generations} 代")
                
                # 自适应参数
                F, CR = self._adaptive_parameters(generation)
                
                # 创建新种群
                new_population = []
                
                for i in range(self.population_size):
                    # 增强变异
                    mutant = self._enhanced_mutation(i, F, generation)
                    
                    # 交叉
                    trial = self._crossover(self.population[i].position, mutant, CR)
                    
                    # 评估
                    trial_fitness = evaluate_individual_fitness((trial, self.bounds_list))
                    
                    # 选择
                    if trial_fitness > self.population[i].fitness:
                        new_individual = Individual(position=trial, fitness=trial_fitness, generation=generation)
                        # 局部搜索增强
                        new_individual = self._local_search(new_individual)
                        new_population.append(new_individual)
                    else:
                        new_population.append(copy.deepcopy(self.population[i]))
                
                # 更新种群
                self.population = new_population
                
                # 更新最佳个体
                self._update_best()
                
                # 子种群迁移
                self._migration(generation)
                
                # 重启机制
                self._restart_mechanism()
                
                # 记录历史
                diversity = self._calculate_diversity()
                self.diversity_history.append(diversity)
                self.fitness_history.append(self.best_fitness)
                
                # 输出信息
                print(f"  参数: F={F:.3f}, CR={CR:.3f}")
                print(f"  最佳适应度: {self.best_fitness:.6f}")
                print(f"  多样性: {diversity:.4f}, 停滞: {self.stagnation_count}, 重启: {self.restart_count}")
                # 每代输出当前最佳解的前三个最长遮蔽区间
                try:
                    if self.best_individual is not None:
                        pos = self.best_individual.position
                        intervals, _dur = compute_intervals_for_params(
                            uav_direction=float(pos[1]),
                            uav_speed=float(pos[0]),
                            smoke_deploy_time=float(pos[2]),
                            smoke_explode_delay=float(pos[3]),
                            algorithm="adaptive"
                        )
                        intervals_sorted = sorted(intervals, key=lambda ab: (ab[1] - ab[0]), reverse=True)
                        top3 = intervals_sorted[:3]
                        if len(top3) > 0:
                            print(f"  前三区间: {top3}")
                        else:
                            print("  前三区间: []")
                except Exception as e:
                    print(f"  区间计算失败: {e}")
                
                # 回调（用于可视化/监控）
                try:
                    if hasattr(self, "_callback") and callable(self._callback):
                        event = {
                            'generation': generation + 1,
                            'F': float(F),
                            'CR': float(CR),
                            'best_fitness': float(self.best_fitness),
                            'best_position': self.best_individual.position.copy() if self.best_individual is not None else None,
                            'diversity': float(diversity),
                            'stagnation_count': int(self.stagnation_count),
                            'restart_count': int(self.restart_count),
                            'population_positions': np.array([ind.position for ind in self.population]),
                            'population_fitness': np.array([ind.fitness for ind in self.population])
                        }
                        self._callback(event)
                except Exception as e:
                    print(f"  回调错误: {e}")
                
                # 检查是否达到目标
                if self.best_fitness >= 4.79:
                    print(f"  🎯 达到目标解！")
                    
                if self.best_fitness >= 4.85:
                    print(f"  🏆 发现超优解！")
                    break
        except KeyboardInterrupt:
            print("\n检测到 Ctrl+C，中断优化并输出当前最优解…")
            if self.best_individual is not None:
                pos = self.best_individual.position
                # 参数标准化，减少浮点抖动
                v = float(np.float64(pos[0]))
                th = float(np.float64(pos[1]))
                td = float(np.float64(pos[2]))
                tf = float(np.float64(pos[3]))

                # 1) 用与优化一致的接口计算时长，确保与适应度一致
                duration_api = calculate_single_uav_single_smoke_masking(
                    uav_direction=th,
                    uav_speed=v,
                    smoke_deploy_time=td,
                    smoke_explode_delay=tf,
                    algorithm="adaptive"
                )

                # 2) 计算区间（先 adaptive，若空且API时长>0则用 smart 兜底）
                intervals, duration_from_intervals = compute_intervals_for_params(
                    uav_direction=th,
                    uav_speed=v,
                    smoke_deploy_time=td,
                    smoke_explode_delay=tf,
                    algorithm="adaptive"
                )
                if len(intervals) == 0 and duration_api > 1e-9:
                    intervals, duration_from_intervals = compute_intervals_for_params(
                        uav_direction=th,
                        uav_speed=v,
                        smoke_deploy_time=td,
                        smoke_explode_delay=tf,
                        algorithm="smart"
                    )

                print("—— 当前最优解 ——")
                print(f"  参数: v={v:.6f}, theta={th:.6f}, t_deploy={td:.6f}, t_fuse={tf:.6f}")
                print(f"  有效遮蔽时长(优化接口): {duration_api:.6f} s")
                print(f"  遮蔽区间数量: {len(intervals)}")
                if len(intervals) > 0:
                    print(f"  区间(前10): {intervals[:10]}")
                    print(f"  区间合计时长: {duration_from_intervals:.6f} s")
                if abs(duration_api - duration_from_intervals) > 1e-3:
                    print(f"  [提示] 时长不一致: 接口={duration_api:.6f}, 区间合计={duration_from_intervals:.6f}")
            else:
                print("尚未产生有效个体。")
            # 直接返回当前最佳
            return (self.best_individual.position if self.best_individual is not None else np.array([])), self.best_fitness
        
        return self.best_individual.position, self.best_fitness
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        j_rand = np.random.randint(0, self.n_dims)
        
        for j in range(self.n_dims):
            if np.random.random() < CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial


def run_parameter_tuning_experiments():
    """运行参数调优实验"""
    print("="*80)
    print("差分进化算法参数调优实验")
    print("="*80)
    
    # 定义不同的参数组合
    param_sets = [
        {
            'name': '保守策略',
            'params': {
                'population_size': 60,
                'max_generations': 600,
                'F_min': 0.4, 'F_max': 0.8,
                'CR_min': 0.1, 'CR_max': 0.7,
                'restart_threshold': 40,
                'local_search_prob': 0.05
            }
        },
        {
            'name': '激进策略',
            'params': {
                'population_size': 100,
                'max_generations': 1000,
                'F_min': 0.2, 'F_max': 1.5,
                'CR_min': 0.05, 'CR_max': 0.95,
                'restart_threshold': 30,
                'local_search_prob': 0.15
            }
        },
        {
            'name': '平衡策略',
            'params': {
                'population_size': 80,
                'max_generations': 800,
                'F_min': 0.3, 'F_max': 1.2,
                'CR_min': 0.05, 'CR_max': 0.95,
                'restart_threshold': 50,
                'local_search_prob': 0.1
            }
        }
    ]
    
    results = {}
    
    for param_set in param_sets:
        print(f"\n{'='*20} {param_set['name']} {'='*20}")
        
        optimizer = TunedDifferentialEvolution(**param_set['params'])
        
        start_time = time.time()
        best_position, best_fitness = optimizer.optimize()
        end_time = time.time()
        
        results[param_set['name']] = {
            'best_fitness': best_fitness,
            'time': end_time - start_time,
            'fitness_history': optimizer.fitness_history
        }
        
        print(f"\n{param_set['name']} 结果:")
        print(f"  最佳适应度: {best_fitness:.6f}")
        print(f"  优化时间: {end_time - start_time:.2f}s")
        print(f"  距离4.8差距: {4.8 - best_fitness:.6f}")
    
    # 找出最佳策略
    best_strategy = max(results.keys(), key=lambda x: results[x]['best_fitness'])
    print(f"\n🏆 最佳策略: {best_strategy}")
    print(f"   最佳适应度: {results[best_strategy]['best_fitness']:.6f}")
    
    return results


def main():
    """主函数"""
    print("问题2：差分进化算法精细调参版本")
    print("目标：从4.588提升到4.8+")
    
    # 运行参数调优实验
    results = run_parameter_tuning_experiments()
    
    return results


if __name__ == "__main__":
    results = main() 