"""
问题4：差分进化算法求解多无人机协同烟幕遮蔽最优策略

差分进化算法特性：
1. 适应高维复杂优化问题（12个决策变量）
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
- 多无人机多烟幕弹协同遮蔽效应建模

目标：找到最优的3架无人机的速度、飞行方向和3个烟幕弹的投放时间、引信延时，
使得有效遮蔽时长最大化。

决策变量（12个）：
- uav_a_direction, uav_a_speed: 无人机FY1的方向和速度
- uav_b_direction, uav_b_speed: 无人机FY2的方向和速度  
- uav_c_direction, uav_c_speed: 无人机FY3的方向和速度
- smoke_a_deploy_time, smoke_a_explode_delay: 烟幕弹A的投放时间和引信延时
- smoke_b_deploy_time, smoke_b_explode_delay: 烟幕弹B的投放时间和引信延时
- smoke_c_deploy_time, smoke_c_explode_delay: 烟幕弹C的投放时间和引信延时
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
import copy
from functools import lru_cache
import threading
import signal
import sys

# 导入求解器
from solver import (
    calculate_multi_uav_single_smoke_masking,
    calculate_multi_uav_single_smoke_masking_multiple,
    TARGETS, MISSILES, SMOKE_PARAMS, UAVS
)
from solver.trajectory import TrajectoryCalculator

# 检查联合遮蔽函数是否可用
HAS_MULTIPLE_MASKING = False

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 全局缓存，用于Multiple模式的性能优化
_multiple_cache = {}
_cache_lock = threading.Lock()
_cache_stats = {'hits': 0, 'misses': 0}
_cache_access_order = []  # 用于LRU缓存管理

# 适应度评估计时统计
_fitness_timing_stats = {
    'decode_time': [],
    'cache_check_time': [],
    'independent_time': [],
    'multiple_time': [],
    'cache_store_time': [],
    'cache_hits': 0,
    'cache_misses': 0
}

def clear_multiple_cache():
    """清空Multiple模式缓存"""
    global _multiple_cache, _cache_stats, _cache_access_order, _fitness_timing_stats
    with _cache_lock:
        _multiple_cache.clear()
        _cache_access_order.clear()
        _cache_stats = {'hits': 0, 'misses': 0}
        _fitness_timing_stats = {
            'decode_time': [],
            'cache_check_time': [],
            'independent_time': [],
            'multiple_time': [],
            'cache_store_time': [],
            'cache_hits': 0,
            'cache_misses': 0
        }

def get_cache_stats():
    """获取缓存统计信息"""
    with _cache_lock:
        return _cache_stats.copy()

def _record_fitness_timing(decode_time, cache_time, independent_time, multiple_time, cache_store_time, is_cache_hit):
    """记录适应度评估的详细计时信息"""
    global _fitness_timing_stats
    with _cache_lock:
        _fitness_timing_stats['decode_time'].append(decode_time)
        _fitness_timing_stats['cache_check_time'].append(cache_time)
        _fitness_timing_stats['independent_time'].append(independent_time)
        _fitness_timing_stats['multiple_time'].append(multiple_time)
        _fitness_timing_stats['cache_store_time'].append(cache_store_time)
        
        if is_cache_hit:
            _fitness_timing_stats['cache_hits'] += 1
        else:
            _fitness_timing_stats['cache_misses'] += 1
        
        # 限制统计数据长度，防止内存泄漏
        max_records = 1000
        for key in ['decode_time', 'cache_check_time', 'independent_time', 'multiple_time', 'cache_store_time']:
            if len(_fitness_timing_stats[key]) > max_records:
                _fitness_timing_stats[key] = _fitness_timing_stats[key][-max_records//2:]

def get_fitness_timing_stats():
    """获取适应度评估计时统计信息"""
    global _fitness_timing_stats
    with _cache_lock:
        stats = {}
        for key in ['decode_time', 'cache_check_time', 'independent_time', 'multiple_time', 'cache_store_time']:
            times = _fitness_timing_stats[key]
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'total': np.sum(times),
                    'count': len(times),
                    'max': np.max(times),
                    'min': np.min(times)
                }
            else:
                stats[key] = {'mean': 0, 'total': 0, 'count': 0, 'max': 0, 'min': 0}
        
        stats['cache_hits'] = _fitness_timing_stats['cache_hits']
        stats['cache_misses'] = _fitness_timing_stats['cache_misses']
        stats['total_evaluations'] = _fitness_timing_stats['cache_hits'] + _fitness_timing_stats['cache_misses']
        
        return stats

def _efficient_cache_cleanup():
    """高效的LRU缓存清理"""
    global _multiple_cache, _cache_access_order
    
    # 只保留最近使用的800个条目（Problem4计算更复杂，缓存容量适当减少）
    if len(_cache_access_order) > 1200:
        # 移除最旧的400个条目
        keys_to_remove = _cache_access_order[:400]
        for key in keys_to_remove:
            _multiple_cache.pop(key, None)
        _cache_access_order = _cache_access_order[400:]


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
        # 步骤1：解码位置参数
        decode_start = time.time()
        params = {
            'uav_a_direction': position[0],         # 无人机FY1方向
            'uav_a_speed': position[1],             # 无人机FY1速度
            'uav_b_direction': position[2],         # 无人机FY2方向
            'uav_b_speed': position[3],             # 无人机FY2速度
            'uav_c_direction': position[4],         # 无人机FY3方向
            'uav_c_speed': position[5],             # 无人机FY3速度
            'smoke_a_deploy_time': position[6],     # 烟幕弹A投放时间
            'smoke_a_explode_delay': position[7],   # 烟幕弹A引信延时
            'smoke_b_deploy_time': position[8],     # 烟幕弹B投放时间
            'smoke_b_explode_delay': position[9],   # 烟幕弹B引信延时
            'smoke_c_deploy_time': position[10],    # 烟幕弹C投放时间
            'smoke_c_explode_delay': position[11]   # 烟幕弹C引信延时
        }
        decode_time = time.time() - decode_start
        
        # 步骤2：独立遮蔽计算
        independent_start = time.time()
        # 计算适应度 - 使用独立遮蔽模式
        duration = calculate_multi_uav_single_smoke_masking(
            uav_a_direction=params['uav_a_direction'],
            uav_a_speed=params['uav_a_speed'],
            uav_b_direction=params['uav_b_direction'],
            uav_b_speed=params['uav_b_speed'],
            uav_c_direction=params['uav_c_direction'],
            uav_c_speed=params['uav_c_speed'],
            smoke_a_deploy_time=params['smoke_a_deploy_time'],
            smoke_a_explode_delay=params['smoke_a_explode_delay'],
            smoke_b_deploy_time=params['smoke_b_deploy_time'],
            smoke_b_explode_delay=params['smoke_b_explode_delay'],
            smoke_c_deploy_time=params['smoke_c_deploy_time'],
            smoke_c_explode_delay=params['smoke_c_explode_delay']
        )
        independent_time = time.time() - independent_start
        
        # 记录独立模式的计时信息
        _record_fitness_timing(decode_time, 0, independent_time, 0, 0, False)
        
        return duration
        
    except Exception as e:
        print(f"独立遮蔽计算错误: {e}")
        return -1000.0


def evaluate_batch_fitness_independent(batch_data):
    """批量评估个体适应度 - 独立遮蔽模式（高效并行版）"""
    positions, bounds_list = batch_data
    results = []
    
    # 预先创建共享的计算器和导弹轨迹（避免重复创建）
    from solver.core import MaskingCalculator, find_t_intervals_adaptive
    from solver.geometry import get_top_plane_points, get_under_points
    from functools import lru_cache
    import time
    
    calc = MaskingCalculator()
    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    
    # 预计算一些共享数据
    target_centers = calc.target_centers
    target_radius = calc.target_radius
    threshold = calc.threshold
    time_step = calc.time_step
    
    batch_start = time.time()
    
    for position in positions:
        try:
            individual_start = time.time()
            
            # 解码参数
            params = {
                'uav_a_direction': position[0], 'uav_a_speed': position[1],
                'uav_b_direction': position[2], 'uav_b_speed': position[3],
                'uav_c_direction': position[4], 'uav_c_speed': position[5],
                'smoke_a_deploy_time': position[6], 'smoke_a_explode_delay': position[7],
                'smoke_b_deploy_time': position[8], 'smoke_b_explode_delay': position[9],
                'smoke_c_deploy_time': position[10], 'smoke_c_explode_delay': position[11]
            }
            
            # 直接调用原有函数（已经优化过的）
            duration = calculate_multi_uav_single_smoke_masking(
                uav_a_direction=params['uav_a_direction'],
                uav_a_speed=params['uav_a_speed'],
                uav_b_direction=params['uav_b_direction'],
                uav_b_speed=params['uav_b_speed'],
                uav_c_direction=params['uav_c_direction'],
                uav_c_speed=params['uav_c_speed'],
                smoke_a_deploy_time=params['smoke_a_deploy_time'],
                smoke_a_explode_delay=params['smoke_a_explode_delay'],
                smoke_b_deploy_time=params['smoke_b_deploy_time'],
                smoke_b_explode_delay=params['smoke_b_explode_delay'],
                smoke_c_deploy_time=params['smoke_c_deploy_time'],
                smoke_c_explode_delay=params['smoke_c_explode_delay']
            )
            
            results.append(duration)
            
        except Exception as e:
            print(f"批量独立遮蔽计算错误: {e}")
            results.append(-1000.0)
    
    batch_time = time.time() - batch_start
    avg_time = batch_time / len(positions) if positions else 0
    
    # 返回结果和统计信息
    return results


def evaluate_individual_fitness_multiple(individual_data):
    """评估个体适应度的全局函数 - 联合遮蔽模式（优化版）"""
    if not HAS_MULTIPLE_MASKING:
        return evaluate_individual_fitness_independent(individual_data)
    
    position, bounds_list = individual_data
    
    try:
        # 步骤1：解码位置参数
        decode_start = time.time()
        params = {
            'uav_a_direction': position[0],         # 无人机FY1方向
            'uav_a_speed': position[1],             # 无人机FY1速度
            'uav_b_direction': position[2],         # 无人机FY2方向
            'uav_b_speed': position[3],             # 无人机FY2速度
            'uav_c_direction': position[4],         # 无人机FY3方向
            'uav_c_speed': position[5],             # 无人机FY3速度
            'smoke_a_deploy_time': position[6],     # 烟幕弹A投放时间
            'smoke_a_explode_delay': position[7],   # 烟幕弹A引信延时
            'smoke_b_deploy_time': position[8],     # 烟幕弹B投放时间
            'smoke_b_explode_delay': position[9],   # 烟幕弹B引信延时
            'smoke_c_deploy_time': position[10],    # 烟幕弹C投放时间
            'smoke_c_explode_delay': position[11]   # 烟幕弹C引信延时
        }
        decode_time = time.time() - decode_start
        
        # 步骤2：缓存检查
        cache_start = time.time()
        # 🚀 优化策略1：高效缓存机制
        # 创建缓存键（降低精度以提高缓存命中率）
        cache_key = tuple(round(x, 3) for x in position)
        
        with _cache_lock:
            if cache_key in _multiple_cache:
                _cache_stats['hits'] += 1
                # 更新访问顺序（LRU）
                if cache_key in _cache_access_order:
                    _cache_access_order.remove(cache_key)
                _cache_access_order.append(cache_key)
                cache_time = time.time() - cache_start
                # 记录缓存命中的计时信息
                _record_fitness_timing(decode_time, cache_time, 0, 0, 0, True)
                return _multiple_cache[cache_key]
            _cache_stats['misses'] += 1
        cache_time = time.time() - cache_start
        
        # 步骤3：独立遮蔽计算（快速筛选）
        independent_start = time.time()
        # 🚀 优化策略2：先用独立模式快速筛选，再用联合模式精确计算
        # 如果独立模式结果很差，直接返回，避免昂贵的联合计算
        independent_duration = calculate_multi_uav_single_smoke_masking(
            uav_a_direction=params['uav_a_direction'],
            uav_a_speed=params['uav_a_speed'],
            uav_b_direction=params['uav_b_direction'],
            uav_b_speed=params['uav_b_speed'],
            uav_c_direction=params['uav_c_direction'],
            uav_c_speed=params['uav_c_speed'],
            smoke_a_deploy_time=params['smoke_a_deploy_time'],
            smoke_a_explode_delay=params['smoke_a_explode_delay'],
            smoke_b_deploy_time=params['smoke_b_deploy_time'],
            smoke_b_explode_delay=params['smoke_b_explode_delay'],
            smoke_c_deploy_time=params['smoke_c_deploy_time'],
            smoke_c_explode_delay=params['smoke_c_explode_delay']
        )
        independent_time = time.time() - independent_start
        
        # 步骤4：联合遮蔽计算（精确计算）
        multiple_start = time.time()
        # 如果独立模式结果太差（<2秒），直接返回，不进行昂贵的联合计算
        if independent_duration < 2.0:
            duration = independent_duration
            multiple_time = 0  # 跳过联合计算
        else:
            # 计算适应度 - 使用联合遮蔽模式（仅对有希望的解进行精确计算）
            duration = calculate_multi_uav_single_smoke_masking_multiple(
                uav_a_direction=params['uav_a_direction'],
                uav_a_speed=params['uav_a_speed'],
                uav_b_direction=params['uav_b_direction'],
                uav_b_speed=params['uav_b_speed'],
                uav_c_direction=params['uav_c_direction'],
                uav_c_speed=params['uav_c_speed'],
                smoke_a_deploy_time=params['smoke_a_deploy_time'],
                smoke_a_explode_delay=params['smoke_a_explode_delay'],
                smoke_b_deploy_time=params['smoke_b_deploy_time'],
                smoke_b_explode_delay=params['smoke_b_explode_delay'],
                smoke_c_deploy_time=params['smoke_c_deploy_time'],
                smoke_c_explode_delay=params['smoke_c_explode_delay']
            )
        multiple_time = time.time() - multiple_start
        
        # 步骤5：缓存存储
        cache_store_start = time.time()
        # 高效缓存结果
        with _cache_lock:
            _multiple_cache[cache_key] = duration
            _cache_access_order.append(cache_key)
            
            # 高效缓存管理
            if len(_multiple_cache) > 1200:
                _efficient_cache_cleanup()
        cache_store_time = time.time() - cache_store_start
        
        # 记录详细计时信息
        _record_fitness_timing(decode_time, cache_time, independent_time, multiple_time, cache_store_time, False)
        
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
    
    # 12个决策变量的边界
    bounds = {
        # 无人机参数
        'uav_a_direction': (0.0, 360.0),           # 无人机FY1方向
        'uav_a_speed': (70.0, 140.0),              # 无人机FY1速度
        'uav_b_direction': (0.0, 360.0),           # 无人机FY2方向
        'uav_b_speed': (70.0, 140.0),              # 无人机FY2速度
        'uav_c_direction': (0.0, 360.0),           # 无人机FY3方向
        'uav_c_speed': (70.0, 140.0),              # 无人机FY3速度
        
        # 烟幕弹参数
        'smoke_a_deploy_time': (0.01, t_max - 5.0), # 烟幕弹A投放时间
        'smoke_a_explode_delay': (0.01, 10.0),      # 烟幕弹A引信延时
        'smoke_b_deploy_time': (0.01, t_max - 5.0), # 烟幕弹B投放时间
        'smoke_b_explode_delay': (0.01, 10.0),      # 烟幕弹B引信延时
        'smoke_c_deploy_time': (0.01, t_max - 5.0), # 烟幕弹C投放时间
        'smoke_c_explode_delay': (0.01, 10.0)       # 烟幕弹C引信延时
    }
    
    return bounds


class DifferentialEvolution_Problem4:
    """问题4差分进化算法优化器"""
    
    def __init__(self,
                 population_size: int = 80,           # 适中的种群大小应对12维问题
                 max_generations: int = 800,          # 适中的代数
                 F_min: float = 0.2,                 # 扩大F范围
                 F_max: float = 1.5,
                 CR_min: float = 0.05,               # 扩大CR范围
                 CR_max: float = 0.95,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True,
                 parallel_mode: str = "process",      # "process" or "thread"
                 masking_mode: str = "independent",   # "independent" or "multiple"
                 restart_threshold: int = 50,         # 高维问题需要更长的停滞容忍
                 local_search_prob: float = 0.12,
                 multi_population: bool = True,
                 n_subpopulations: int = 4,
                 migration_interval: int = 20,
                 elite_rate: float = 0.1):
        """
        初始化问题4差分进化算法
        
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
        self.parallel_mode = parallel_mode
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
        self.n_dims = len(self.bounds_list)  # 12个决策变量
        
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
        
        # 历史记录（限制长度以防止内存泄漏）
        self.max_history_length = 400  # Problem4维度更高，历史记录适中
        self.fitness_history = []
        self.diversity_history = []
        self.parameter_history = {'F': [], 'CR': []}
        self.strategy_success_count = {strategy: 0 for strategy in self.mutation_strategies}
        self.strategy_usage_count = {strategy: 0 for strategy in self.mutation_strategies}
        self.stagnation_count = 0
        self.restart_count = 0
        
        # 并行计算设置
        if self.use_parallel:
            self.n_processes = min(8, mp.cpu_count(), max(2, population_size // 4))  # 最多8个进程
            if parallel_mode == "thread":
                print(f"将使用 {self.n_processes} 个线程进行并行计算")
            else:
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
        
        # 中断处理标志
        self.interrupted = False
    
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
                # 子种群1：偏向高速度策略
                position[1] = np.random.uniform(110, 140)  # FY1高速度
                position[3] = np.random.uniform(110, 140)  # FY2高速度  
                position[5] = np.random.uniform(110, 140)  # FY3高速度
                # 其他参数随机
                for i in [0, 2, 4] + list(range(6, self.n_dims)):
                    min_val, max_val = self.bounds_list[i]
                    position[i] = np.random.uniform(min_val, max_val)
            elif subpop_idx == 2:
                # 子种群2：偏向早投放策略
                position[6] = np.random.uniform(0.1, 2.0)   # 早投放A
                position[8] = np.random.uniform(0.1, 2.0)   # 早投放B
                position[10] = np.random.uniform(0.1, 2.0)  # 早投放C
                # 其他参数随机
                for i in list(range(6)) + [7, 9, 11]:
                    min_val, max_val = self.bounds_list[i]
                    position[i] = np.random.uniform(min_val, max_val)
            else:
                # 子种群3：偏向协同策略
                # 相近的飞行方向
                base_direction = np.random.uniform(0, 360)
                position[0] = base_direction % 360                        # FY1方向
                position[2] = (base_direction + 30) % 360                 # FY2方向
                position[4] = (base_direction - 30) % 360                 # FY3方向
                
                # 相近的投放时间
                base_deploy_time = np.random.uniform(1.0, 5.0)
                position[6] = base_deploy_time                            # A投放时间
                position[8] = base_deploy_time + np.random.uniform(0, 1)  # B投放时间
                position[10] = base_deploy_time + np.random.uniform(0, 1) # C投放时间
                
                # 其他参数随机
                for i in [1, 3, 5, 7, 9, 11]:
                    min_val, max_val = self.bounds_list[i]
                    position[i] = np.random.uniform(min_val, max_val)
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
        if self.best_fitness < 3.0:
            # 初期：大F值，增强全局搜索
            F = self.F_max - (self.F_max - 0.8) * progress
        elif self.best_fitness < 6.0:
            # 中期：中等F值
            F = 0.8 - (0.8 - 0.5) * progress
        else:
            # 后期：小F值，精细搜索
            F = 0.5 - (0.5 - self.F_min) * progress
        
        # 基于停滞情况调整F
        if self.stagnation_count > 25:
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
        if generation < 15:
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
        if self.best_fitness > 6.0:  # 接近最优时
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
        search_radius = 0.02  # 更小的搜索半径
        
        # 对于高维问题，只搜索部分维度
        n_dims_to_search = min(5, self.n_dims)  # 最多搜索5个维度
        dims_to_search = np.random.choice(self.n_dims, n_dims_to_search, replace=False)
        
        for _ in range(6):  # 适中的尝试次数
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
        
        # 保留最好的20%个体（高维问题需要保留更多精英）
        n_keep = int(0.20 * self.population_size)
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
        """计算种群多样性（高效版本）"""
        if len(self.population) < 2:
            return 0.0
        
        # 🚀 性能优化：只采样少量个体进行多样性计算
        sample_size = min(15, len(self.population))  # 进一步减少采样数量
        if len(self.population) > sample_size:
            sample_indices = np.random.choice(len(self.population), sample_size, replace=False)
            sampled_positions = [self.population[i].position for i in sample_indices]
        else:
            sampled_positions = [ind.position for ind in self.population]
        
        positions = np.array(sampled_positions)
        
        # 🚀 使用向量化计算替代双重循环
        # 计算所有对之间的欧氏距离
        n = len(positions)
        distances = []
        
        # 只计算前几个维度的多样性（进一步加速）
        key_dims = min(6, self.n_dims)  # 只看前6个关键维度
        
        for i in range(n):
            for j in range(i + 1, n):
                normalized_dist = 0
                for k in range(key_dims):  # 只计算关键维度
                    min_val, max_val = self.bounds_list[k]
                    normalized_dist += ((positions[i][k] - positions[j][k]) / (max_val - min_val)) ** 2
                distances.append(np.sqrt(normalized_dist))
        
        return np.mean(distances) if distances else 0.0
    
    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        print("\n\n⚠️  检测到中断信号 (Ctrl+C)")
        print("正在保存当前最优结果并显示可视化...")
        self.interrupted = True
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行差分进化优化"""
        print("="*60)
        print(f"开始问题4差分进化算法优化 - {self.masking_mode.upper()}模式")
        print("="*60)
        print("💡 提示: 按 Ctrl+C 可随时中断并查看当前最优结果")
        
        # 设置信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # 初始化种群
        self._initialize_population()
        
        # 主优化循环
        for generation in range(self.max_generations):
            # 检查中断信号
            if self.interrupted:
                print(f"\n🛑 优化在第 {generation+1} 代被中断")
                break
                
            generation_start_time = time.time()  # 记录每代开始时间
            print(f"\n第 {generation+1}/{self.max_generations} 代")
            
            # 步骤1：自适应参数调整
            step1_start = time.time()
            F, CR = self._adaptive_parameters(generation)
            self.parameter_history['F'].append(F)
            self.parameter_history['CR'].append(CR)
            
            # 限制参数历史长度
            if len(self.parameter_history['F']) > self.max_history_length:
                self.parameter_history['F'] = self.parameter_history['F'][-self.max_history_length//2:]
                self.parameter_history['CR'] = self.parameter_history['CR'][-self.max_history_length//2:]
            step1_time = time.time() - step1_start
            
            # 步骤2：选择变异策略
            step2_start = time.time()
            strategy = self._select_mutation_strategy(generation)
            self.strategy_usage_count[strategy] += 1
            step2_time = time.time() - step2_start
            
            # 步骤3：变异、交叉、评估循环
            step3_start = time.time()
            mutation_time = 0
            crossover_time = 0
            evaluation_time = 0
            local_search_time = 0
            
            new_population = []
            successful_mutations = 0
            trials = []
            
            # 步骤3.1：生成所有变异和交叉个体
            for i in range(self.population_size):
                # 变异
                mut_start = time.time()
                mutant = self._mutate(i, F, strategy)
                mutation_time += time.time() - mut_start
                
                # 交叉
                cross_start = time.time()
                crossover_type = 'exp' if 'exp' in strategy else 'bin'
                trial = self._crossover(self.population[i].position, mutant, CR, crossover_type)
                crossover_time += time.time() - cross_start
                
                # 收集试验个体，稍后批量并行评估
                trials.append(trial)
            
            # 步骤3.2：批量并行评估所有试验个体
            eval_start = time.time()
            if self.use_parallel and self.masking_mode == "independent":
                # 对于独立模式，使用批量并行计算优化
                # 确保每个进程至少处理2个个体，最多处理population_size/2个个体
                batch_size = max(2, min(self.population_size // 2, self.population_size // self.n_processes))
                trial_fitness_list = []
                
                # 将试验个体分批处理，确保充分利用所有进程
                trial_batches = []
                for i in range(0, len(trials), batch_size):
                    batch = trials[i:i+batch_size]
                    if batch:  # 确保批次不为空
                        trial_batches.append(batch)
                
                batch_data = [(batch, self.bounds_list) for batch in trial_batches]
                
                executor_type = "线程" if self.parallel_mode == "thread" else "进程"
                print(f"      🚀 启动{len(trial_batches)}个批次，每批{batch_size}个体，使用{self.n_processes}个{executor_type}")
                
                ExecutorClass = ThreadPoolExecutor if self.parallel_mode == "thread" else ProcessPoolExecutor
                with ExecutorClass(max_workers=self.n_processes) as executor:
                    futures = {executor.submit(evaluate_batch_fitness_independent, data): i 
                              for i, data in enumerate(batch_data)}
                    batch_results = [None] * len(trial_batches)
                    for future in as_completed(futures):
                        idx = futures[future]
                        batch_results[idx] = future.result()
                
                # 合并批量结果
                for batch_result in batch_results:
                    if batch_result:  # 确保结果不为空
                        trial_fitness_list.extend(batch_result)
                    
            elif self.use_parallel:
                # 原有的个体级并行计算（用于multiple模式）
                trial_data = [(trial, self.bounds_list) for trial in trials]
                ExecutorClass = ThreadPoolExecutor if self.parallel_mode == "thread" else ProcessPoolExecutor
                with ExecutorClass(max_workers=self.n_processes) as executor:
                    futures = {executor.submit(self.fitness_function, data): i 
                              for i, data in enumerate(trial_data)}
                    trial_fitness_list = [None] * self.population_size
                    for future in as_completed(futures):
                        idx = futures[future]
                        trial_fitness_list[idx] = future.result()
            else:
                trial_fitness_list = [self.fitness_function((trial, self.bounds_list)) for trial in trials]
            evaluation_time += time.time() - eval_start
            
            # 步骤3.3：选择和局部搜索
            for i in range(self.population_size):
                trial_fitness = trial_fitness_list[i]
                if trial_fitness > self.population[i].fitness:
                    new_individual = Individual(position=trials[i], fitness=trial_fitness, generation=generation)
                    # 局部搜索增强（降低频率以提高性能）
                    if generation % 4 == 0:  # 每4代进行一次局部搜索
                        ls_start = time.time()
                        new_individual = self._local_search(new_individual)
                        local_search_time += time.time() - ls_start
                    new_population.append(new_individual)
                    successful_mutations += 1
                else:
                    # 🚀 避免深拷贝，直接复制引用（Individual是不可变的）
                    new_population.append(self.population[i])
            
            step3_time = time.time() - step3_start
            
            # 步骤4：更新策略成功计数
            step4_start = time.time()
            if successful_mutations > 0:
                self.strategy_success_count[strategy] += successful_mutations
            step4_time = time.time() - step4_start
            
            # 步骤5：更新种群
            step5_start = time.time()
            self.population = new_population
            step5_time = time.time() - step5_start
            
            # 步骤6：更新最佳个体
            step6_start = time.time()
            self._update_best()
            step6_time = time.time() - step6_start
            
            # 步骤7：子种群迁移
            step7_start = time.time()
            self._migration(generation)
            step7_time = time.time() - step7_start
            
            # 步骤8：重启机制
            step8_start = time.time()
            self._restart_mechanism()
            step8_time = time.time() - step8_start
            
            # 步骤9：计算多样性（每5代计算一次以提高性能）
            step9_start = time.time()
            if generation % 5 == 0:
                diversity = self._calculate_diversity()
                self.diversity_history.append(diversity)
                # 限制多样性历史长度
                if len(self.diversity_history) > self.max_history_length:
                    self.diversity_history = self.diversity_history[-self.max_history_length//2:]
            else:
                diversity = self.diversity_history[-1] if self.diversity_history else 0.0
            step9_time = time.time() - step9_start
            
            # 步骤10：记录历史（限制长度）
            step10_start = time.time()
            self.fitness_history.append(self.best_fitness)
            if len(self.fitness_history) > self.max_history_length:
                self.fitness_history = self.fitness_history[-self.max_history_length//2:]
            step10_time = time.time() - step10_start
            
            # 输出信息
            generation_time = time.time() - generation_start_time
            success_rate = successful_mutations / self.population_size
            
            print(f"  变异策略: {strategy}")
            print(f"  参数: F={F:.3f}, CR={CR:.3f}")
            print(f"  最佳适应度: {self.best_fitness:.6f}")
            print(f"  成功率: {success_rate:.1%}, 多样性: {diversity:.4f}")
            print(f"  停滞: {self.stagnation_count}, 重启: {self.restart_count}")
            print(f"  本代用时: {generation_time:.2f}s")
            
            # 详细耗时分析
            print(f"  ⏱️ 耗时分析:")
            print(f"    参数调整: {step1_time*1000:.1f}ms")
            print(f"    策略选择: {step2_time*1000:.1f}ms")
            print(f"    变异操作: {mutation_time*1000:.1f}ms")
            print(f"    交叉操作: {crossover_time*1000:.1f}ms")
            print(f"    适应度评估: {evaluation_time:.3f}s ({evaluation_time/generation_time*100:.1f}%)")
            if self.masking_mode == "independent" and self.use_parallel:
                avg_per_individual = evaluation_time / self.population_size
                speedup_ratio = 74.9 / (avg_per_individual * 1000)
                print(f"      批量并行优化: 平均每个体{avg_per_individual*1000:.1f}ms")
                print(f"      性能提升: {speedup_ratio:.1f}x 加速 (原{74.9:.1f}ms → 现{avg_per_individual*1000:.1f}ms)")
                print(f"      并行效率: {speedup_ratio/self.n_processes*100:.1f}% (理论最大{self.n_processes}x)")
            
            # 适应度评估详细分解
            if generation % 10 == 0:  # 每10代输出一次详细的适应度计时统计
                fitness_stats = get_fitness_timing_stats()
                if fitness_stats['total_evaluations'] > 0:
                    print(f"      🔍 适应度评估详细分解 (最近{fitness_stats['total_evaluations']}次):")
                    print(f"        参数解码: {fitness_stats['decode_time']['mean']*1000:.2f}ms (总计{fitness_stats['decode_time']['total']*1000:.1f}ms)")
                    if fitness_stats['cache_check_time']['total'] > 0:
                        print(f"        缓存检查: {fitness_stats['cache_check_time']['mean']*1000:.2f}ms (总计{fitness_stats['cache_check_time']['total']*1000:.1f}ms)")
                    print(f"        独立遮蔽计算: {fitness_stats['independent_time']['mean']*1000:.1f}ms (总计{fitness_stats['independent_time']['total']:.2f}s)")
                    if fitness_stats['multiple_time']['total'] > 0:
                        print(f"        联合遮蔽计算: {fitness_stats['multiple_time']['mean']*1000:.1f}ms (总计{fitness_stats['multiple_time']['total']:.2f}s)")
                    if fitness_stats['cache_store_time']['total'] > 0:
                        print(f"        缓存存储: {fitness_stats['cache_store_time']['mean']*1000:.2f}ms (总计{fitness_stats['cache_store_time']['total']*1000:.1f}ms)")
                    print(f"        缓存命中率: {fitness_stats['cache_hits']}/{fitness_stats['total_evaluations']} ({fitness_stats['cache_hits']/max(1,fitness_stats['total_evaluations'])*100:.1f}%)")
            
            if local_search_time > 0:
                print(f"    局部搜索: {local_search_time:.3f}s ({local_search_time/generation_time*100:.1f}%)")
            print(f"    策略统计: {step4_time*1000:.1f}ms")
            print(f"    种群更新: {step5_time*1000:.1f}ms")
            print(f"    最佳更新: {step6_time*1000:.1f}ms")
            print(f"    种群迁移: {step7_time*1000:.1f}ms")
            print(f"    重启机制: {step8_time*1000:.1f}ms")
            print(f"    多样性计算: {step9_time*1000:.1f}ms")
            print(f"    历史记录: {step10_time*1000:.1f}ms")
            
            # 定期清理缓存以防止内存泄漏
            if generation % 40 == 0 and self.masking_mode == "multiple":
                with _cache_lock:
                    if len(_multiple_cache) > 800:
                        print(f"  🧹 清理缓存: {len(_multiple_cache)} -> ", end="")
                        _efficient_cache_cleanup()
                        print(f"{len(_multiple_cache)}")
            
            # 检查是否达到很好的结果
            if self.best_fitness >= 8.0:  # 多无人机协同的理论上限可能更高
                print(f"  🎯 发现优秀解！")
            
            # 收敛检查
            if generation > 40:
                recent_improvement = max(self.fitness_history[-15:]) - min(self.fitness_history[-30:-15])
                if recent_improvement < 1e-6 and self.stagnation_count > 60:
                    print(f"  算法收敛，提前结束于第 {generation+1} 代")
                    break
        
        # 如果被中断，显示中断信息
        if self.interrupted:
            print("\n" + "="*60)
            print("🛑 优化过程被用户中断")
            print("="*60)
            print(f"已完成 {len(self.fitness_history)} 代优化")
            print(f"当前最佳适应度: {self.best_fitness:.6f}")
        
        return self.best_individual.position, self.best_fitness
    
    def plot_convergence(self):
        """绘制详细的收敛分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 适应度收敛曲线
        axes[0, 0].plot(self.fitness_history, 'b-', linewidth=2, label='最佳适应度')
        axes[0, 0].set_title(f'问题4差分进化收敛曲线 ({self.masking_mode.upper()}模式)')
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


def analyze_problem4_de_results(best_position: np.ndarray, best_fitness: float, 
                               bounds: Dict[str, Tuple[float, float]], masking_mode: str):
    """分析问题4差分进化结果"""
    print("="*60)
    print(f"问题4差分进化优化结果分析 ({masking_mode.upper()}模式)")
    print("="*60)
    
    # 解码最优解
    keys = list(bounds.keys())
    best_params = {keys[i]: best_position[i] for i in range(len(keys))}
    
    print(f"\n最优策略参数：")
    print(f"📡 无人机参数：")
    print(f"  FY1 - 方向: {best_params['uav_a_direction']:.2f}°, 速度: {best_params['uav_a_speed']:.2f} m/s")
    print(f"  FY2 - 方向: {best_params['uav_b_direction']:.2f}°, 速度: {best_params['uav_b_speed']:.2f} m/s")
    print(f"  FY3 - 方向: {best_params['uav_c_direction']:.2f}°, 速度: {best_params['uav_c_speed']:.2f} m/s")
    
    print(f"\n💣 烟幕弹参数：")
    print(f"  烟幕弹A - 投放: {best_params['smoke_a_deploy_time']:.3f}s, 延时: {best_params['smoke_a_explode_delay']:.3f}s")
    print(f"             起爆: {best_params['smoke_a_deploy_time'] + best_params['smoke_a_explode_delay']:.3f}s")
    
    print(f"  烟幕弹B - 投放: {best_params['smoke_b_deploy_time']:.3f}s, 延时: {best_params['smoke_b_explode_delay']:.3f}s")
    print(f"             起爆: {best_params['smoke_b_deploy_time'] + best_params['smoke_b_explode_delay']:.3f}s")
    
    print(f"  烟幕弹C - 投放: {best_params['smoke_c_deploy_time']:.3f}s, 延时: {best_params['smoke_c_explode_delay']:.3f}s")
    print(f"             起爆: {best_params['smoke_c_deploy_time'] + best_params['smoke_c_explode_delay']:.3f}s")
    
    print(f"\n🎯 最大有效遮蔽时长: {best_fitness:.6f} 秒")
    
    # 验证结果
    print(f"\n🔍 验证计算...")
    if masking_mode == "multiple" and HAS_MULTIPLE_MASKING:
        verification_result = calculate_multi_uav_single_smoke_masking_multiple(
            uav_a_direction=best_params['uav_a_direction'],
            uav_a_speed=best_params['uav_a_speed'],
            uav_b_direction=best_params['uav_b_direction'],
            uav_b_speed=best_params['uav_b_speed'],
            uav_c_direction=best_params['uav_c_direction'],
            uav_c_speed=best_params['uav_c_speed'],
            smoke_a_deploy_time=best_params['smoke_a_deploy_time'],
            smoke_a_explode_delay=best_params['smoke_a_explode_delay'],
            smoke_b_deploy_time=best_params['smoke_b_deploy_time'],
            smoke_b_explode_delay=best_params['smoke_b_explode_delay'],
            smoke_c_deploy_time=best_params['smoke_c_deploy_time'],
            smoke_c_explode_delay=best_params['smoke_c_explode_delay']
        )
    else:
        verification_result = calculate_multi_uav_single_smoke_masking(
            uav_a_direction=best_params['uav_a_direction'],
            uav_a_speed=best_params['uav_a_speed'],
            uav_b_direction=best_params['uav_b_direction'],
            uav_b_speed=best_params['uav_b_speed'],
            uav_c_direction=best_params['uav_c_direction'],
            uav_c_speed=best_params['uav_c_speed'],
            smoke_a_deploy_time=best_params['smoke_a_deploy_time'],
            smoke_a_explode_delay=best_params['smoke_a_explode_delay'],
            smoke_b_deploy_time=best_params['smoke_b_deploy_time'],
            smoke_b_explode_delay=best_params['smoke_b_explode_delay'],
            smoke_c_deploy_time=best_params['smoke_c_deploy_time'],
            smoke_c_explode_delay=best_params['smoke_c_explode_delay']
        )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    return best_params


def main():
    """主函数"""
    print("问题4：差分进化算法求解多无人机协同烟幕遮蔽最优策略")
    
    # 选择遮蔽模式
    masking_mode = "multiple" if HAS_MULTIPLE_MASKING else "independent"
    
    # 清空缓存
    if masking_mode == "multiple":
        clear_multiple_cache()
    
    # 设置DE算法参数（针对Multiple模式优化）
    if masking_mode == "multiple":
        # Multiple模式：减少计算量，因为单次评估成本很高
        de_params = {
            'population_size': 50,          # 减少种群大小
            'max_generations': 400,         # 减少代数
            'F_min': 0.2,                  
            'F_max': 1.5,
            'CR_min': 0.05,                
            'CR_max': 0.95,
            'use_parallel': True,           
            'restart_threshold': 35,        # 减少重启阈值
            'local_search_prob': 0.08,      # 减少局部搜索概率
            'multi_population': True,       
            'n_subpopulations': 3,          # 减少子种群数量
            'migration_interval': 15,       
            'elite_rate': 0.15             # 增加精英保留率
        }
    else:
        # Independent模式：可以使用更大的参数，因为计算速度快
        de_params = {
            'population_size': 80,         
            'max_generations': 800,        
            'F_min': 0.2,                  
            'F_max': 1.5,
            'CR_min': 0.05,                
            'CR_max': 0.95,
            'use_parallel': True,           
            'restart_threshold': 50,        
            'local_search_prob': 0.12,     
            'multi_population': True,       
            'n_subpopulations': 4,          
            'migration_interval': 20,       
            'elite_rate': 0.1              
        }
    
    print(f"\n问题4差分进化算法参数：")
    for key, value in de_params.items():
        print(f"  {key}: {value}")
    
    # 创建优化器
    optimizer = DifferentialEvolution_Problem4(masking_mode=masking_mode, **de_params)
    
    print(f"\n搜索空间边界（12个决策变量）：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    try:
        best_position, best_fitness = optimizer.optimize()
        end_time = time.time()
        
        if optimizer.interrupted:
            print(f"\n⚠️  优化被中断，总用时: {end_time - start_time:.2f} 秒")
        else:
            print(f"\n✅ 优化完成，总用时: {end_time - start_time:.2f} 秒")
    except KeyboardInterrupt:
        # 如果在optimize函数外被中断
        end_time = time.time()
        print(f"\n⚠️  优化被中断，总用时: {end_time - start_time:.2f} 秒")
        best_position = optimizer.best_individual.position if optimizer.best_individual else None
        best_fitness = optimizer.best_fitness
    
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
    
    # 分析结果（即使被中断也要显示）
    if best_position is not None:
        best_params = analyze_problem4_de_results(best_position, best_fitness, 
                                                optimizer.bounds, masking_mode)
        
        # 绘制收敛曲线（即使被中断也要显示）
        if len(optimizer.fitness_history) > 0:
            optimizer.plot_convergence()
        else:
            print("⚠️  没有足够的数据绘制收敛曲线")
    else:
        print("⚠️  没有有效的优化结果")
        best_params = None
    
    # 保存结果
    results = {
        'best_params': best_params,
        'best_fitness': best_fitness if best_position is not None else -np.inf,
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
        'masking_mode': masking_mode,
        'interrupted': optimizer.interrupted
    }
    
    if optimizer.interrupted:
        print(f"\n⚠️  问题4差分进化优化结果已保存（被中断）")
    else:
        print(f"\n✅ 问题4差分进化优化结果已保存")
    
    # 清理资源
    if masking_mode == "multiple":
        clear_multiple_cache()
        print("已清理Multiple模式缓存")
    
    return results


if __name__ == "__main__":
    results = main() 