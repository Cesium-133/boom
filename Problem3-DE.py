"""
问题3：差分进化算法求解单无人机三烟幕弹最优策略

差分进化算法特性：
1. 适应高维复杂优化问题（8个决策变量）
2. 支持两种遮蔽计算模式：独立遮蔽 vs 联合遮蔽
3. 多种变异策略动态选择
4. 自适应参数调整机制
5. 多子种群并行搜索
6. 重启和局部搜索增强
7. 自适应步长区间查找算法

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
from functools import lru_cache
import threading

# 导入求解器
from solver import (
    calculate_single_uav_triple_smoke_masking,
    calculate_single_uav_triple_smoke_masking_multiple,
    TARGETS, MISSILES, SMOKE_PARAMS
)
from solver.trajectory import TrajectoryCalculator

HAS_MULTIPLE_MASKING = True

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局缓存，用于Multiple模式的性能优化
_multiple_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {'hits': 0, 'misses': 0}

def clear_multiple_cache():
    """清空Multiple模式缓存"""
    global _multiple_cache, _cache_stats
    with _cache_lock:
        _multiple_cache.clear()
        _cache_stats = {'hits': 0, 'misses': 0}

def get_cache_stats():
    """获取缓存统计信息"""
    with _cache_lock:
        return _cache_stats.copy()


@dataclass
class Individual:
    """个体类"""
    position: np.ndarray
    fitness: float
    generation: int = 0


# 全局函数，用于并行计算
def evaluate_individual_fitness_independent(individual_data):
    """评估个体适应度的全局函数 - 独立遮蔽模式"""
    position, bounds_list = individual_data
    
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


def evaluate_individual_fitness_multiple(individual_data):
    """评估个体适应度的全局函数 - 联合遮蔽模式（优化版）"""
    if not HAS_MULTIPLE_MASKING:
        return evaluate_individual_fitness_independent(individual_data)
    
    position, bounds_list = individual_data
    
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
        
        # 🚀 优化策略1：缓存机制
        # 创建缓存键（降低精度以提高缓存命中率）
        cache_key = tuple(round(x, 3) for x in position)
        
        with _cache_lock:
            if cache_key in _multiple_cache:
                _cache_stats['hits'] += 1
                return _multiple_cache[cache_key]
            _cache_stats['misses'] += 1
        
        # 🚀 优化策略2：先用独立模式快速筛选，再用联合模式精确计算
        # 如果独立模式结果很差，直接返回，避免昂贵的联合计算
        independent_duration = calculate_single_uav_triple_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_a_deploy_time=params['smoke_a_deploy_time'],
            smoke_a_explode_delay=params['smoke_a_explode_delay'],
            smoke_b_deploy_delay=params['smoke_b_deploy_delay'],
            smoke_b_explode_delay=params['smoke_b_explode_delay'],
            smoke_c_deploy_delay=params['smoke_c_deploy_delay'],
            smoke_c_explode_delay=params['smoke_c_explode_delay'],
            algorithm="adaptive"  # 使用自适应算法
        )
        
        # 如果独立模式结果太差（<3秒），直接返回，不进行昂贵的联合计算
        if independent_duration < 3.0:
            duration = independent_duration
        else:
            # 计算适应度 - 使用联合遮蔽模式（仅对有希望的解进行精确计算）
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
        
        # 缓存结果
        with _cache_lock:
            _multiple_cache[cache_key] = duration
            # 限制缓存大小
            if len(_multiple_cache) > 2000:
                # 清除最旧的1000个条目
                keys_to_remove = list(_multiple_cache.keys())[:1000]
                for key in keys_to_remove:
                    del _multiple_cache[key]
        
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


class DifferentialEvolution_Problem3:
    """问题3差分进化算法优化器"""
    
    def __init__(self,
                 population_size: int = 100,          # 增大种群以应对高维问题
                 max_generations: int = 1000,         # 增加代数
                 F_min: float = 0.2,                 # 扩大F范围
                 F_max: float = 1.5,
                 CR_min: float = 0.05,               # 扩大CR范围
                 CR_max: float = 0.95,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True,
                 masking_mode: str = "independent",   # "independent" or "multiple"
                 restart_threshold: int = 60,         # 高维问题需要更长的停滞容忍
                 local_search_prob: float = 0.15,
                 multi_population: bool = True,
                 n_subpopulations: int = 4,
                 migration_interval: int = 25,
                 elite_rate: float = 0.1):
        """
        初始化问题3差分进化算法
        
        Args:
            masking_mode: 遮蔽计算模式
                - "independent": 独立遮蔽（任一烟幕弹满足条件即可）
                - "multiple": 联合遮蔽（考虑多烟幕弹协同效应）
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.use_parallel = use_parallel
        self.masking_mode = masking_mode
        self.restart_threshold = restart_threshold
        self.local_search_prob = local_search_prob
        self.multi_population = multi_population
        self.n_subpopulations = n_subpopulations
        self.migration_interval = migration_interval
        self.elite_rate = elite_rate
        
        # 设置适应度评估函数
        if masking_mode == "multiple" and HAS_MULTIPLE_MASKING:
            self.fitness_function = evaluate_individual_fitness_multiple
            print("🔗 使用联合遮蔽模式")
        else:
            self.fitness_function = evaluate_individual_fitness_independent
            print("🔸 使用独立遮蔽模式")
        
        # 设置边界
        if bounds is None:
            self.bounds = calculate_bounds()
        else:
            self.bounds = bounds
            
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)  # 8个决策变量
        
        # 针对高维问题的变异策略
        self.mutation_strategies = [
            'DE/rand/1',
            'DE/best/1', 
            'DE/current-to-best/1',
            'DE/rand/2',
            'DE/best/2',
            'DE/rand-to-best/1',
            'DE/current-to-rand/1',
            'DE/best/1/exp',          # 指数交叉
            'DE/rand/1/bin'           # 二项式交叉
        ]
        
        # 多子种群设置
        if multi_population:
            self.subpop_size = population_size // n_subpopulations
            self.subpopulations = []
        else:
            self.subpop_size = population_size
        
        # 初始化
        self.population = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # 历史记录
        self.fitness_history = []
        self.diversity_history = []
        self.parameter_history = {'F': [], 'CR': []}
        self.strategy_success_count = {strategy: 0 for strategy in self.mutation_strategies}
        self.strategy_usage_count = {strategy: 0 for strategy in self.mutation_strategies}
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
                    position = self._generate_initial_position(sub_idx)
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
                position = self._generate_initial_position()
                individual = Individual(position=position, fitness=-np.inf)
                self.population.append(individual)
        
        # 评估初始种群
        self._evaluate_population()
        self._update_best()
        
        print(f"初始化完成，最佳适应度: {self.best_fitness:.6f}")
    
    def _generate_initial_position(self, subpop_idx: int = 0) -> np.ndarray:
        """生成初始位置（支持多子种群的不同初始化策略）"""
        position = np.zeros(self.n_dims)
        
        if self.multi_population:
            # 不同子种群使用不同的初始化策略
            if subpop_idx == 0:
                # 子种群0：完全随机
                for i, (min_val, max_val) in enumerate(self.bounds_list):
                    position[i] = np.random.uniform(min_val, max_val)
            elif subpop_idx == 1:
                # 子种群1：偏向高速度
                position[0] = np.random.uniform(110, 140)  # 高速度
                position[1] = np.random.uniform(0, 360)    # 随机方向
                for i in range(2, self.n_dims):
                    min_val, max_val = self.bounds_list[i]
                    position[i] = np.random.uniform(min_val, max_val)
            elif subpop_idx == 2:
                # 子种群2：偏向早投放
                position[0] = np.random.uniform(70, 140)   # 随机速度
                position[1] = np.random.uniform(0, 360)    # 随机方向
                position[2] = np.random.uniform(0.1, 2.0)  # 早投放
                for i in range(3, self.n_dims):
                    min_val, max_val = self.bounds_list[i]
                    position[i] = np.random.uniform(min_val, max_val)
            else:
                # 子种群3：偏向密集投放
                position[0] = np.random.uniform(70, 140)   # 随机速度
                position[1] = np.random.uniform(0, 360)    # 随机方向
                position[2] = np.random.uniform(0.1, 5.0)  # 随机投放时间
                position[3] = np.random.uniform(0.1, 3.0)  # 较短延时
                position[4] = np.random.uniform(0.1, 2.0)  # 短间隔
                position[5] = np.random.uniform(0.1, 3.0)  # 较短延时
                position[6] = np.random.uniform(0.1, 2.0)  # 短间隔
                position[7] = np.random.uniform(0.1, 3.0)  # 较短延时
        else:
            # 单种群：完全随机初始化
            for i, (min_val, max_val) in enumerate(self.bounds_list):
                position[i] = np.random.uniform(min_val, max_val)
        
        return position
    
    def _evaluate_population(self):
        """评估种群适应度"""
        if self.use_parallel:
            individual_data = [(ind.position, self.bounds_list) for ind in self.population]
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {executor.submit(self.fitness_function, data): i 
                          for i, data in enumerate(individual_data)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    fitness_value = future.result()
                    self.population[idx].fitness = fitness_value
        else:
            for individual in self.population:
                fitness_value = self.fitness_function((individual.position, self.bounds_list))
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
        """高维问题的自适应参数调整"""
        progress = generation / self.max_generations
        
        # 基于当前最佳适应度的F调整
        if self.best_fitness < 5.0:
            # 初期：大F值，增强全局搜索
            F = self.F_max - (self.F_max - 0.8) * progress
        elif self.best_fitness < 8.0:
            # 中期：中等F值
            F = 0.8 - (0.8 - 0.5) * progress
        else:
            # 后期：小F值，精细搜索
            F = 0.5 - (0.5 - self.F_min) * progress
        
        # 基于停滞情况调整F
        if self.stagnation_count > 30:
            F = min(self.F_max, F * 1.3)  # 增强探索
        
        # 高维问题的CR调整
        diversity = self._calculate_diversity()
        if diversity < 0.05:  # 多样性很低
            CR = self.CR_max  # 最大交叉概率
        elif diversity < 0.15:  # 多样性低
            CR = self.CR_min + (self.CR_max - self.CR_min) * 0.8
        elif diversity > 0.4:   # 多样性高
            CR = self.CR_min + (self.CR_max - self.CR_min) * 0.3
        else:
            # 正常情况
            CR = self.CR_min + (self.CR_max - self.CR_min) * (1 - progress)
        
        return F, CR
    
    def _select_mutation_strategy(self, generation: int) -> str:
        """选择变异策略（考虑高维问题特点）"""
        if generation < 20:
            # 前期：偏向探索性策略
            strategies = ['DE/rand/1', 'DE/rand/2', 'DE/current-to-rand/1']
            return np.random.choice(strategies)
        
        # 基于成功率选择策略
        success_rates = {}
        for strategy in self.mutation_strategies:
            usage = self.strategy_usage_count[strategy]
            success = self.strategy_success_count[strategy]
            if usage > 0:
                success_rates[strategy] = success / usage
            else:
                success_rates[strategy] = 0.3  # 默认成功率
        
        # 对于高维问题，给予某些策略额外权重
        if self.best_fitness > 8.0:  # 接近最优时
            success_rates['DE/best/1'] *= 1.2
            success_rates['DE/current-to-best/1'] *= 1.2
        
        # 轮盘赌选择
        strategies = list(success_rates.keys())
        probabilities = list(success_rates.values())
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(strategies, p=probabilities)
    
    def _mutate(self, target_idx: int, F: float, strategy: str) -> np.ndarray:
        """变异操作（适配高维问题）"""
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
        
        else:
            # 默认策略
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position))
        
        # 边界处理
        for i in range(self.n_dims):
            min_val, max_val = self.bounds_list[i]
            mutant[i] = np.clip(mutant[i], min_val, max_val)
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float, strategy: str = 'bin') -> np.ndarray:
        """交叉操作（支持二项式和指数交叉）"""
        trial = target.copy()
        
        if strategy == 'bin' or 'bin' in strategy:
            # 二项式交叉
            j_rand = np.random.randint(0, self.n_dims)
            for j in range(self.n_dims):
                if np.random.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
        
        elif strategy == 'exp' or 'exp' in strategy:
            # 指数交叉
            n = np.random.randint(0, self.n_dims)
            L = 0
            while True:
                trial[(n + L) % self.n_dims] = mutant[(n + L) % self.n_dims]
                L += 1
                if L >= self.n_dims or np.random.random() >= CR:
                    break
        
        return trial
    
    def _local_search(self, individual: Individual) -> Individual:
        """针对高维问题的局部搜索"""
        if np.random.random() > self.local_search_prob:
            return individual
        
        best_local = copy.deepcopy(individual)
        search_radius = 0.03  # 更小的搜索半径
        
        # 对于高维问题，只搜索部分维度
        n_dims_to_search = min(4, self.n_dims)  # 最多搜索4个维度
        dims_to_search = np.random.choice(self.n_dims, n_dims_to_search, replace=False)
        
        for _ in range(8):  # 增加尝试次数
            new_position = individual.position.copy()
            
            # 只在选定的维度上进行扰动
            for dim in dims_to_search:
                min_val, max_val = self.bounds_list[dim]
                range_val = max_val - min_val
                perturbation = np.random.normal(0, search_radius * range_val)
                new_position[dim] = np.clip(new_position[dim] + perturbation, min_val, max_val)
            
            # 评估新位置
            fitness = self.fitness_function((new_position, self.bounds_list))
            
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
        """重启机制（适配高维问题）"""
        if self.stagnation_count < self.restart_threshold:
            return
        
        print(f"    触发重启机制 (停滞{self.stagnation_count}代)")
        
        # 保留最好的25%个体（高维问题需要保留更多精英）
        n_keep = int(0.25 * self.population_size)
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        elite = self.population[:n_keep]
        
        # 重新初始化其余个体
        for i in range(n_keep, self.population_size):
            if self.multi_population:
                subpop_idx = i // self.subpop_size
                position = self._generate_initial_position(subpop_idx)
            else:
                position = self._generate_initial_position()
            self.population[i] = Individual(position=position, fitness=-np.inf)
        
        # 重新评估
        self._evaluate_population()
        self.restart_count += 1
        self.stagnation_count = 0
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性（高维版本）"""
        if len(self.population) < 2:
            return 0.0
        
        positions = np.array([ind.position for ind in self.population])
        
        # 对于高维问题，计算部分维度的多样性以减少计算量
        sample_size = min(50, len(positions))  # 最多采样50个个体
        if len(positions) > sample_size:
            sample_indices = np.random.choice(len(positions), sample_size, replace=False)
            positions = positions[sample_indices]
        
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
        """执行差分进化优化"""
        print("="*60)
        print(f"开始问题3差分进化算法优化 - {self.masking_mode.upper()}模式")
        print("="*60)
        
        # 初始化种群
        self._initialize_population()
        
        # 主优化循环
        for generation in range(self.max_generations):
            print(f"\n第 {generation+1}/{self.max_generations} 代")
            
            # 自适应参数
            F, CR = self._adaptive_parameters(generation)
            self.parameter_history['F'].append(F)
            self.parameter_history['CR'].append(CR)
            
            # 选择变异策略
            strategy = self._select_mutation_strategy(generation)
            self.strategy_usage_count[strategy] += 1
            
            # 创建新种群
            new_population = []
            successful_mutations = 0
            
            for i in range(self.population_size):
                # 变异
                mutant = self._mutate(i, F, strategy)
                
                # 交叉
                crossover_type = 'exp' if 'exp' in strategy else 'bin'
                trial = self._crossover(self.population[i].position, mutant, CR, crossover_type)
                
                # 评估试验个体
                trial_fitness = self.fitness_function((trial, self.bounds_list))
                
                # 选择
                if trial_fitness > self.population[i].fitness:
                    new_individual = Individual(position=trial, fitness=trial_fitness, generation=generation)
                    # 局部搜索增强
                    new_individual = self._local_search(new_individual)
                    new_population.append(new_individual)
                    successful_mutations += 1
                else:
                    new_population.append(copy.deepcopy(self.population[i]))
            
            # 更新策略成功计数
            if successful_mutations > 0:
                self.strategy_success_count[strategy] += successful_mutations
            
            # 更新种群
            self.population = new_population
            
            # 更新最佳个体
            self._update_best()
            
            # 子种群迁移
            self._migration(generation)
            
            # 重启机制
            self._restart_mechanism()
            
            # 计算多样性
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # 记录历史
            self.fitness_history.append(self.best_fitness)
            
            # 输出信息
            success_rate = successful_mutations / self.population_size
            
            print(f"  变异策略: {strategy}")
            print(f"  参数: F={F:.3f}, CR={CR:.3f}")
            print(f"  最佳适应度: {self.best_fitness:.6f}")
            print(f"  成功率: {success_rate:.1%}, 多样性: {diversity:.4f}")
            print(f"  停滞: {self.stagnation_count}, 重启: {self.restart_count}")
            
            # 检查是否达到很好的结果
            if self.best_fitness >= 10.0:  # 三烟幕弹的理论上限可能更高
                print(f"  🎯 发现优秀解！")
            
            # 收敛检查
            if generation > 50:
                recent_improvement = max(self.fitness_history[-20:]) - min(self.fitness_history[-40:-20])
                if recent_improvement < 1e-6 and self.stagnation_count > 80:
                    print(f"  算法收敛，提前结束于第 {generation+1} 代")
                    break
        
        return self.best_individual.position, self.best_fitness
    
    def plot_convergence(self):
        """绘制详细的收敛分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 适应度收敛曲线
        axes[0, 0].plot(self.fitness_history, 'b-', linewidth=2, label='最佳适应度')
        axes[0, 0].set_title(f'问题3差分进化收敛曲线 ({self.masking_mode.upper()}模式)')
        axes[0, 0].set_xlabel('代数')
        axes[0, 0].set_ylabel('适应度（有效遮蔽时长）')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 多样性变化
        axes[0, 1].plot(self.diversity_history, 'g-', linewidth=2)
        axes[0, 1].set_title('种群多样性变化')
        axes[0, 1].set_xlabel('代数')
        axes[0, 1].set_ylabel('多样性')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 参数变化
        axes[1, 0].plot(self.parameter_history['F'], 'r-', linewidth=2, label='F (变异因子)')
        axes[1, 0].plot(self.parameter_history['CR'], 'orange', linewidth=2, label='CR (交叉概率)')
        axes[1, 0].set_title('自适应参数变化')
        axes[1, 0].set_xlabel('代数')
        axes[1, 0].set_ylabel('参数值')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 变异策略使用统计
        strategies = list(self.strategy_usage_count.keys())
        usage_counts = list(self.strategy_usage_count.values())
        success_counts = [self.strategy_success_count[s] for s in strategies]
        
        x = np.arange(len(strategies))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, usage_counts, width, label='使用次数', alpha=0.8)
        axes[1, 1].bar(x + width/2, success_counts, width, label='成功次数', alpha=0.8)
        axes[1, 1].set_title('变异策略统计')
        axes[1, 1].set_xlabel('变异策略')
        axes[1, 1].set_ylabel('次数')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([s.replace('DE/', '').replace('/', '/\n') for s in strategies], rotation=45, fontsize=8)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_problem3_de_results(best_position: np.ndarray, best_fitness: float, 
                               bounds: Dict[str, Tuple[float, float]], masking_mode: str):
    """分析问题3差分进化结果"""
    print("="*60)
    print(f"问题3差分进化优化结果分析 ({masking_mode.upper()}模式)")
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
            algorithm="adaptive"
        )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    return best_params


def main():
    """主函数"""
    print("问题3：差分进化算法求解单无人机三烟幕弹最优策略")
    
    # 选择遮蔽模式
    masking_mode = "multiple" if HAS_MULTIPLE_MASKING else "independent"
    
    # 清空缓存
    if masking_mode == "multiple":
        clear_multiple_cache()
    
    # 设置DE算法参数（针对Multiple模式优化）
    if masking_mode == "multiple":
        # Multiple模式：减少计算量，因为单次评估成本很高
        de_params = {
            'population_size': 60,          # 减少种群大小
            'max_generations': 300,         # 减少代数
            'F_min': 0.2,                  
            'F_max': 1.5,
            'CR_min': 0.05,                
            'CR_max': 0.95,
            'use_parallel': True,           
            'restart_threshold': 40,        # 减少重启阈值
            'local_search_prob': 0.08,      # 减少局部搜索概率
            'multi_population': True,       
            'n_subpopulations': 3,          # 减少子种群数量
            'migration_interval': 20,       
            'elite_rate': 0.15             # 增加精英保留率
        }
    else:
        # Independent模式：可以使用更大的参数，因为计算速度快
        de_params = {
            'population_size': 100,         
            'max_generations': 1000,        
            'F_min': 0.2,                  
            'F_max': 1.5,
            'CR_min': 0.05,                
            'CR_max': 0.95,
            'use_parallel': True,           
            'restart_threshold': 60,        
            'local_search_prob': 0.15,     
            'multi_population': True,       
            'n_subpopulations': 4,          
            'migration_interval': 25,       
            'elite_rate': 0.1              
        }
    
    print(f"\n问题3差分进化算法参数：")
    for key, value in de_params.items():
        print(f"  {key}: {value}")
    
    # 创建优化器
    optimizer = DifferentialEvolution_Problem3(masking_mode=masking_mode, **de_params)
    
    print(f"\n搜索空间边界（8个决策变量）：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    best_position, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    
    # 性能统计
    if masking_mode == "multiple":
        cache_stats = get_cache_stats()
        total_calls = cache_stats['hits'] + cache_stats['misses']
        hit_rate = cache_stats['hits'] / max(1, total_calls) * 100
        print(f"\nMultiple模式性能统计:")
        print(f"  总函数调用次数: {total_calls}")
        print(f"  缓存命中次数: {cache_stats['hits']}")
        print(f"  缓存命中率: {hit_rate:.1f}%")
        print(f"  缓存大小: {len(_multiple_cache)}")
    
    # 分析结果
    best_params = analyze_problem3_de_results(best_position, best_fitness, 
                                            optimizer.bounds, masking_mode)
    
    # 绘制收敛曲线
    optimizer.plot_convergence()
    
    # 保存结果
    results = {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'de_params': de_params,
        'bounds': optimizer.bounds,
        'fitness_history': optimizer.fitness_history,
        'diversity_history': optimizer.diversity_history,
        'parameter_history': optimizer.parameter_history,
        'strategy_statistics': {
            'usage_count': optimizer.strategy_usage_count,
            'success_count': optimizer.strategy_success_count
        },
        'restart_count': optimizer.restart_count,
        'masking_mode': masking_mode
    }
    
    print(f"\n问题3差分进化优化结果已保存")
    
    # 清理资源
    if masking_mode == "multiple":
        clear_multiple_cache()
        print("已清理Multiple模式缓存")
    
    return results


if __name__ == "__main__":
    results = main() 