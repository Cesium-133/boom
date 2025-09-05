"""
问题2：差分进化算法求解最优无人机策略

差分进化算法特性：
1. 简单高效的全局优化算法
2. 自适应变异和交叉操作
3. 多种变异策略选择
4. 并行计算加速
5. 自适应步长区间查找算法
6. 参数自适应调整

核心优化：
- 使用Numba JIT编译加速核心几何计算
- 采用自适应步长算法优化时间区间查找
- LRU缓存减少重复计算
- 多种变异策略动态选择

目标：找到最优的无人机速度、飞行方向、烟幕弹投放时间和引信延时，
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
from solver import calculate_single_uav_single_smoke_masking, TARGETS, MISSILES
from solver.trajectory import TrajectoryCalculator

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


@dataclass
class Individual:
    """个体类"""
    position: np.ndarray
    fitness: float
    generation: int = 0


# 全局函数，用于并行计算
def evaluate_individual_fitness(individual_data):
    """评估个体适应度的全局函数 - 使用自适应步长算法"""
    position, bounds_list = individual_data
    
    try:
        # 解码位置
        params = {
            'v_FY1': position[0],
            'theta_FY1': position[1],
            't_deploy': position[2],
            't_fuse': position[3]
        }
        
        # 计算适应度 - 使用自适应步长算法
        duration = calculate_single_uav_single_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_deploy_time=params['t_deploy'],
            smoke_explode_delay=params['t_fuse'],
            algorithm="adaptive"  # 使用自适应步长算法
        )
        
        return duration
        
    except Exception as e:
        print(f"计算错误: {e}")
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
    
    bounds = {
        'v_FY1': (70.0, 140.0),
        'theta_FY1': (0.0, 360.0),
        't_deploy': (0.1, t_max - 2.0),
        't_fuse': (0.1, 10.0)
    }
    
    return bounds


class DifferentialEvolution:
    """差分进化算法优化器"""
    
    def __init__(self,
                 population_size: int = 50,
                 max_generations: int = 200,
                 F_min: float = 0.4,
                 F_max: float = 0.9,
                 CR_min: float = 0.1,
                 CR_max: float = 0.9,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True,
                 mutation_strategies: List[str] = None,
                 adaptive_parameters: bool = True,
                 elite_rate: float = 0.1):
        """
        初始化差分进化算法
        
        Args:
            population_size: 种群大小
            max_generations: 最大代数
            F_min, F_max: 变异因子范围
            CR_min, CR_max: 交叉概率范围
            bounds: 搜索边界
            use_parallel: 是否使用并行计算
            mutation_strategies: 变异策略列表
            adaptive_parameters: 是否使用自适应参数
            elite_rate: 精英保留率
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.F_min = F_min
        self.F_max = F_max
        self.CR_min = CR_min
        self.CR_max = CR_max
        self.use_parallel = use_parallel
        self.adaptive_parameters = adaptive_parameters
        self.elite_rate = elite_rate
        
        # 设置边界
        if bounds is None:
            self.bounds = calculate_bounds()
        else:
            self.bounds = bounds
            
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)
        
        # 变异策略
        if mutation_strategies is None:
            self.mutation_strategies = [
                'DE/rand/1',
                'DE/best/1', 
                'DE/current-to-best/1',
                'DE/rand/2',
                'DE/best/2'
            ]
        else:
            self.mutation_strategies = mutation_strategies
        
        # 初始化种群
        self.population = []
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # 历史记录
        self.fitness_history = []
        self.diversity_history = []
        self.parameter_history = {'F': [], 'CR': []}
        self.strategy_success_count = {strategy: 0 for strategy in self.mutation_strategies}
        self.strategy_usage_count = {strategy: 0 for strategy in self.mutation_strategies}
        
        # 并行计算设置
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), population_size)
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
    
    def _initialize_population(self):
        """初始化种群"""
        print(f"初始化种群，大小: {self.population_size}")
        
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
        
        # 找到最佳个体
        self._update_best()
        
        print(f"初始化完成，最佳适应度: {self.best_fitness:.6f}")
    
    def _evaluate_population(self):
        """评估种群适应度"""
        if self.use_parallel:
            # 并行计算
            individual_data = [(ind.position, self.bounds_list) for ind in self.population]
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                futures = {executor.submit(evaluate_individual_fitness, data): i 
                          for i, data in enumerate(individual_data)}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    fitness_value = future.result()
                    self.population[idx].fitness = fitness_value
        else:
            # 串行计算
            for individual in self.population:
                fitness_value = evaluate_individual_fitness((individual.position, self.bounds_list))
                individual.fitness = fitness_value
    
    def _update_best(self):
        """更新最佳个体"""
        for individual in self.population:
            if individual.fitness > self.best_fitness:
                self.best_fitness = individual.fitness
                self.best_individual = copy.deepcopy(individual)
    
    def _adaptive_parameters(self, generation: int) -> Tuple[float, float]:
        """自适应参数调整"""
        if not self.adaptive_parameters:
            # 固定参数
            F = (self.F_min + self.F_max) / 2
            CR = (self.CR_min + self.CR_max) / 2
            return F, CR
        
        # 基于代数的自适应调整
        progress = generation / self.max_generations
        
        # F参数：前期大，后期小（增强全局搜索能力，后期精细搜索）
        F = self.F_max - (self.F_max - self.F_min) * progress
        
        # CR参数：根据种群多样性调整
        diversity = self._calculate_diversity()
        if diversity > 0.5:  # 多样性高时，减少交叉
            CR = self.CR_min + (self.CR_max - self.CR_min) * 0.3
        else:  # 多样性低时，增加交叉
            CR = self.CR_min + (self.CR_max - self.CR_min) * 0.8
        
        # 基于成功率的微调
        if len(self.fitness_history) > 10:
            recent_improvement = max(self.fitness_history[-5:]) - max(self.fitness_history[-10:-5])
            if recent_improvement < 1e-6:  # 改进缓慢，增加探索
                F = min(self.F_max, F * 1.1)
                CR = max(self.CR_min, CR * 0.9)
        
        return F, CR
    
    def _select_mutation_strategy(self, generation: int) -> str:
        """选择变异策略"""
        if generation < 10:
            # 前期随机选择
            return np.random.choice(self.mutation_strategies)
        
        # 基于成功率选择策略
        success_rates = {}
        for strategy in self.mutation_strategies:
            usage = self.strategy_usage_count[strategy]
            success = self.strategy_success_count[strategy]
            if usage > 0:
                success_rates[strategy] = success / usage
            else:
                success_rates[strategy] = 0.5  # 默认成功率
        
        # 轮盘赌选择
        strategies = list(success_rates.keys())
        probabilities = list(success_rates.values())
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(strategies, p=probabilities)
    
    def _mutate(self, target_idx: int, F: float, strategy: str) -> np.ndarray:
        """变异操作"""
        population_indices = list(range(len(self.population)))
        population_indices.remove(target_idx)
        
        if strategy == 'DE/rand/1':
            # DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position))
        
        elif strategy == 'DE/best/1':
            # DE/best/1: v = x_best + F * (x_r1 - x_r2)
            r1, r2 = np.random.choice(population_indices, 2, replace=False)
            mutant = (self.best_individual.position + 
                     F * (self.population[r1].position - self.population[r2].position))
        
        elif strategy == 'DE/current-to-best/1':
            # DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
            r1, r2 = np.random.choice(population_indices, 2, replace=False)
            mutant = (self.population[target_idx].position + 
                     F * (self.best_individual.position - self.population[target_idx].position) +
                     F * (self.population[r1].position - self.population[r2].position))
        
        elif strategy == 'DE/rand/2':
            # DE/rand/2: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
            r1, r2, r3, r4, r5 = np.random.choice(population_indices, 5, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position) +
                     F * (self.population[r4].position - self.population[r5].position))
        
        elif strategy == 'DE/best/2':
            # DE/best/2: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
            r1, r2, r3, r4 = np.random.choice(population_indices, 4, replace=False)
            mutant = (self.best_individual.position + 
                     F * (self.population[r1].position - self.population[r2].position) +
                     F * (self.population[r3].position - self.population[r4].position))
        
        else:
            # 默认使用 DE/rand/1
            r1, r2, r3 = np.random.choice(population_indices, 3, replace=False)
            mutant = (self.population[r1].position + 
                     F * (self.population[r2].position - self.population[r3].position))
        
        # 边界处理
        for i in range(self.n_dims):
            min_val, max_val = self.bounds_list[i]
            mutant[i] = np.clip(mutant[i], min_val, max_val)
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        
        # 确保至少有一个维度来自变异向量
        j_rand = np.random.randint(0, self.n_dims)
        
        for j in range(self.n_dims):
            if np.random.random() < CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def _calculate_diversity(self) -> float:
        """计算种群多样性"""
        if len(self.population) < 2:
            return 0.0
        
        positions = np.array([ind.position for ind in self.population])
        
        # 计算所有个体间的平均距离
        distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                # 归一化距离
                normalized_dist = 0
                for k in range(self.n_dims):
                    min_val, max_val = self.bounds_list[k]
                    normalized_dist += ((positions[i][k] - positions[j][k]) / (max_val - min_val)) ** 2
                distances.append(np.sqrt(normalized_dist))
        
        return np.mean(distances) if distances else 0.0
    
    def _elite_preservation(self, new_population: List[Individual]):
        """精英保留策略"""
        n_elite = max(1, int(self.population_size * self.elite_rate))
        
        # 按适应度排序
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        new_population.sort(key=lambda x: x.fitness, reverse=True)
        
        # 保留最好的精英个体
        elite_individuals = self.population[:n_elite]
        
        # 用精英个体替换新种群中最差的个体
        final_population = new_population[:-n_elite] + elite_individuals
        
        return final_population
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行差分进化优化"""
        print("="*60)
        print("开始差分进化算法优化")
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
                trial = self._crossover(self.population[i].position, mutant, CR)
                
                # 评估试验个体
                trial_fitness = evaluate_individual_fitness((trial, self.bounds_list))
                
                # 选择
                if trial_fitness > self.population[i].fitness:
                    new_individual = Individual(position=trial, fitness=trial_fitness, generation=generation)
                    new_population.append(new_individual)
                    successful_mutations += 1
                else:
                    new_population.append(copy.deepcopy(self.population[i]))
            
            # 更新策略成功计数
            if successful_mutations > 0:
                self.strategy_success_count[strategy] += successful_mutations
            
            # 精英保留
            if self.elite_rate > 0:
                new_population = self._elite_preservation(new_population)
            
            # 更新种群
            self.population = new_population
            
            # 更新最佳个体
            prev_best = self.best_fitness
            self._update_best()
            
            # 计算多样性
            diversity = self._calculate_diversity()
            self.diversity_history.append(diversity)
            
            # 记录历史
            self.fitness_history.append(self.best_fitness)
            
            # 输出信息
            improvement = self.best_fitness - prev_best
            success_rate = successful_mutations / self.population_size
            
            print(f"  变异策略: {strategy}")
            print(f"  参数: F={F:.3f}, CR={CR:.3f}")
            print(f"  最佳适应度: {self.best_fitness:.6f} (改进: {improvement:+.6f})")
            print(f"  成功率: {success_rate:.1%}, 多样性: {diversity:.4f}")
            
            # 检查是否达到很好的结果
            if self.best_fitness >= 4.5:
                print(f"  🎯 发现优秀解！")
            
            # 收敛检查
            if generation > 20:
                recent_improvement = max(self.fitness_history[-10:]) - min(self.fitness_history[-20:-10])
                if recent_improvement < 1e-6:
                    print(f"  算法收敛，提前结束于第 {generation+1} 代")
                    break
        
        return self.best_individual.position, self.best_fitness
    
    def plot_convergence(self):
        """绘制收敛分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 适应度收敛曲线
        axes[0, 0].plot(self.fitness_history, 'b-', linewidth=2, label='最佳适应度')
        axes[0, 0].set_title('差分进化算法收敛曲线')
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
        axes[1, 1].set_xticklabels([s.replace('DE/', '') for s in strategies], rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_de_results(best_position: np.ndarray, best_fitness: float, 
                      bounds: Dict[str, Tuple[float, float]]):
    """分析差分进化结果"""
    print("="*60)
    print("差分进化算法优化结果分析")
    print("="*60)
    
    # 解码最优解
    keys = list(bounds.keys())
    best_params = {keys[i]: best_position[i] for i in range(len(keys))}
    
    print(f"\n最优策略参数：")
    print(f"  无人机速度 (v_FY1): {best_params['v_FY1']:.2f} m/s")
    print(f"  无人机方向 (θ_FY1): {best_params['theta_FY1']:.2f}°")
    print(f"  烟幕弹投放时间 (t_deploy): {best_params['t_deploy']:.3f} s")
    print(f"  烟幕弹引信延时 (t_fuse): {best_params['t_fuse']:.3f} s")
    print(f"  烟幕弹起爆时间: {best_params['t_deploy'] + best_params['t_fuse']:.3f} s")
    
    print(f"\n最大有效遮蔽时长: {best_fitness:.6f} 秒")
    
    # 验证结果
    print(f"\n验证计算...")
    verification_result = calculate_single_uav_single_smoke_masking(
        uav_direction=best_params['theta_FY1'],
        uav_speed=best_params['v_FY1'],
        smoke_deploy_time=best_params['t_deploy'],
        smoke_explode_delay=best_params['t_fuse'],
        algorithm="adaptive"
    )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    return best_params


def main():
    """主函数"""
    print("问题2：差分进化算法求解最优无人机策略")
    
    # 设置DE算法参数
    de_params = {
        'population_size': 50,          # 种群大小
        'max_generations': 500,         # 最大代数
        'F_min': 0.4,                  # 最小变异因子
        'F_max': 0.9,                  # 最大变异因子
        'CR_min': 0.1,                 # 最小交叉概率
        'CR_max': 0.9,                 # 最大交叉概率
        'use_parallel': True,           # 使用并行计算
        'adaptive_parameters': True,    # 自适应参数
        'elite_rate': 0.1              # 精英保留率
    }
    
    print(f"\n差分进化算法参数：")
    for key, value in de_params.items():
        print(f"  {key}: {value}")
    
    # 创建优化器
    optimizer = DifferentialEvolution(**de_params)
    
    print(f"\n搜索空间边界：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    best_position, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    
    # 分析结果
    best_params = analyze_de_results(best_position, best_fitness, optimizer.bounds)
    
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
        }
    }
    
    print(f"\n差分进化优化结果已保存")
    
    return results


if __name__ == "__main__":
    results = main() 