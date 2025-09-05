"""
问题2：使用遗传算法求解最优无人机策略 (交叉验证PSO结果)

目标：找到最优的无人机速度、飞行方向、烟幕弹投放时间和引信延时，
使得有效遮蔽时长最大化。

决策变量：
- v_FY1: 无人机速度 [70, 140] m/s
- θ_FY1: 无人机飞行方向 [0, 360] 度
- t_deploy: 烟幕弹投放时间 [0, T_max] s
- t_fuse: 烟幕弹引信相对时间 [0, T_fuse_max] s

遗传算法特点：
- 基于生物进化原理
- 全局搜索能力强
- 不容易陷入局部最优
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 导入求解器
from solver import calculate_single_uav_single_smoke_masking, TARGETS, MISSILES
from solver.trajectory import TrajectoryCalculator

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 全局函数，用于并行计算
def evaluate_individual(individual_data):
    """评估个体适应度的全局函数"""
    individual, bounds_list = individual_data
    
    try:
        # 解码个体
        params = {
            'v_FY1': individual[0],
            'theta_FY1': individual[1],
            't_deploy': individual[2],
            't_fuse': individual[3]
        }
        
        # 计算适应度
        duration = calculate_single_uav_single_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_deploy_time=params['t_deploy'],
            smoke_explode_delay=params['t_fuse']
        )
        
        return duration
        
    except Exception as e:
        print(f"计算错误: {e}")
        return -1000.0


def calculate_bounds():
    """计算搜索空间边界"""
    print("正在计算搜索空间边界...")
    
    # 计算导弹到达虚假目标的时间
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
        't_deploy': (0.1, t_max - 1.0),
        't_fuse': (0.1, 10.0)
    }
    
    return bounds


class GA_Optimizer:
    """遗传算法优化器"""
    
    def __init__(self,
                 population_size: int = 50,
                 max_generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 elite_size: int = 2,
                 tournament_size: int = 3,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 use_parallel: bool = True):
        """
        初始化遗传算法优化器
        
        Args:
            population_size: 种群大小
            max_generations: 最大代数
            crossover_rate: 交叉概率
            mutation_rate: 变异概率
            elite_size: 精英个体数量
            tournament_size: 锦标赛选择的个体数
            bounds: 变量边界
            use_parallel: 是否使用并行计算
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        self.use_parallel = use_parallel
        
        # 设置边界
        if bounds is None:
            self.bounds = calculate_bounds()
        else:
            self.bounds = bounds
        
        # 转换为列表格式便于索引
        self.bounds_list = list(self.bounds.values())
        self.n_vars = len(self.bounds_list)
        
        # 种群和适应度
        self.population = []
        self.fitness = []
        
        # 历史记录
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # 全局最优
        self.best_individual = None
        self.best_fitness = -np.inf
        
        # 进程池
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), self.population_size)
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
    
    def _create_individual(self) -> List[float]:
        """创建一个随机个体"""
        individual = []
        for i, (min_val, max_val) in enumerate(self.bounds_list):
            individual.append(random.uniform(min_val, max_val))
        return individual
    
    def _initialize_population(self):
        """初始化种群"""
        print("初始化种群...")
        
        self.population = []
        for i in range(self.population_size):
            individual = self._create_individual()
            self.population.append(individual)
        
        # 计算初始适应度
        self._evaluate_population()
        
        print(f"种群初始化完成，最佳适应度: {self.best_fitness:.6f}")
    
    def _evaluate_population(self):
        """评估种群适应度"""
        if self.use_parallel:
            # 并行计算 - 修复顺序问题
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                # 准备数据
                tasks = [(individual, self.bounds_list) for individual in self.population]
                
                # 提交任务并保持顺序
                futures = {executor.submit(evaluate_individual, task): i for i, task in enumerate(tasks)}
                
                # 收集结果，保持顺序
                self.fitness = [0.0] * len(self.population)
                for future in as_completed(futures):
                    idx = futures[future]
                    fitness_value = future.result()
                    self.fitness[idx] = fitness_value
        else:
            # 串行计算
            self.fitness = []
            for i, individual in enumerate(self.population):
                fitness_value = evaluate_individual((individual, self.bounds_list))
                self.fitness.append(fitness_value)
                print(f"个体 {i+1}/{self.population_size} 评估完成，适应度: {fitness_value:.4f}")
        
        # 更新全局最优
        max_idx = np.argmax(self.fitness)
        if self.fitness[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness[max_idx]
            self.best_individual = self.population[max_idx].copy()
    
    def _tournament_selection(self) -> List[float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(self.population_size), self.tournament_size)
        tournament_fitness = [self.fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """模拟二进制交叉(SBX)"""
        eta = 2.0  # 分布指数
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(self.n_vars):
            if random.random() <= self.crossover_rate:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    # 计算交叉参数
                    u = random.random()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    
                    # 生成子代
                    child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i])
                    child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i])
                    
                    # 边界处理
                    min_val, max_val = self.bounds_list[i]
                    child1[i] = np.clip(child1[i], min_val, max_val)
                    child2[i] = np.clip(child2[i], min_val, max_val)
        
        return child1, child2
    
    def _mutate(self, individual: List[float]) -> List[float]:
        """多项式变异"""
        eta = 20.0  # 分布指数
        
        mutated = individual.copy()
        
        for i in range(self.n_vars):
            if random.random() <= self.mutation_rate:
                min_val, max_val = self.bounds_list[i]
                
                # 计算变异参数
                u = random.random()
                if u < 0.5:
                    delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
                else:
                    delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                
                # 应用变异
                mutated[i] = individual[i] + delta * (max_val - min_val) * 0.1
                
                # 边界处理
                mutated[i] = np.clip(mutated[i], min_val, max_val)
        
        return mutated
    
    def _create_next_generation(self):
        """创建下一代种群"""
        new_population = []
        
        # 精英保留
        elite_indices = np.argsort(self.fitness)[-self.elite_size:]
        for idx in elite_indices:
            new_population.append(self.population[idx].copy())
        
        # 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # 交叉
            child1, child2 = self._crossover(parent1, parent2)
            
            # 变异
            child1 = self._mutate(child1)
            child2 = self._mutate(child2)
            
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)
        
        # 更新种群
        self.population = new_population[:self.population_size]
    
    def optimize(self) -> Tuple[List[float], float]:
        """执行遗传算法优化"""
        print("="*60)
        print("开始遗传算法优化")
        print("="*60)
        
        # 初始化种群
        self._initialize_population()
        
        # 进化循环
        for generation in range(self.max_generations):
            print(f"\n第 {generation+1}/{self.max_generations} 代")
            
            # 创建新一代
            self._create_next_generation()
            
            # 评估新种群
            self._evaluate_population()
            
            # 记录历史
            self.best_fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(self.fitness))
            
            # 输出信息
            print(f"  当前最佳适应度: {self.best_fitness:.6f}")
            print(f"  当前平均适应度: {np.mean(self.fitness):.6f}")
            print(f"  种群适应度标准差: {np.std(self.fitness):.6f}")
            
            # 收敛检查
            if generation > 20:
                recent_improvement = max(self.best_fitness_history[-10:]) - min(self.best_fitness_history[-10:])
                if recent_improvement < 1e-6:
                    print(f"  算法收敛，提前结束于第 {generation+1} 代")
                    break
        
        return self.best_individual, self.best_fitness
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        # 子图1：适应度进化
        plt.subplot(2, 2, 1)
        plt.plot(self.best_fitness_history, 'r-', linewidth=2, label='最佳适应度')
        plt.plot(self.avg_fitness_history, 'b-', linewidth=2, label='平均适应度')
        plt.title('GA算法收敛曲线')
        plt.xlabel('代数')
        plt.ylabel('适应度（有效遮蔽时长）')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 子图2：适应度分布
        plt.subplot(2, 2, 2)
        plt.hist(self.fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('当前种群适应度分布')
        plt.xlabel('适应度')
        plt.ylabel('个体数量')
        plt.grid(True, alpha=0.3)
        
        # 子图3：收敛速度
        plt.subplot(2, 2, 3)
        if len(self.best_fitness_history) > 1:
            improvements = np.diff(self.best_fitness_history)
            plt.plot(improvements, 'g-', linewidth=2)
            plt.title('适应度改进速度')
            plt.xlabel('代数')
            plt.ylabel('适应度提升')
            plt.grid(True, alpha=0.3)
        
        # 子图4：种群多样性
        plt.subplot(2, 2, 4)
        diversity = [np.std(self.fitness)]
        plt.plot(diversity, 'orange', marker='o', linewidth=2)
        plt.title('种群多样性')
        plt.xlabel('代数')
        plt.ylabel('适应度标准差')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


def analyze_ga_results(best_individual: List[float], best_fitness: float, bounds: Dict[str, Tuple[float, float]]):
    """分析遗传算法结果"""
    print("="*60)
    print("遗传算法优化结果分析")
    print("="*60)
    
    # 解码最优解
    keys = list(bounds.keys())
    best_params = {keys[i]: best_individual[i] for i in range(len(keys))}
    
    print(f"\n最优策略参数（遗传算法）：")
    print(f"  无人机速度 (v_FY1): {best_params['v_FY1']:.2f} m/s")
    print(f"  无人机方向 (θ_FY1): {best_params['theta_FY1']:.2f}°")
    print(f"  投放时间 (t_deploy): {best_params['t_deploy']:.3f} s")
    print(f"  引信延时 (t_fuse): {best_params['t_fuse']:.3f} s")
    print(f"  起爆时间: {best_params['t_deploy'] + best_params['t_fuse']:.3f} s")
    
    print(f"\n最大有效遮蔽时长: {best_fitness:.6f} 秒")
    
    # 验证结果
    print(f"\n验证计算...")
    verification_result = calculate_single_uav_single_smoke_masking(
        uav_direction=best_params['theta_FY1'],
        uav_speed=best_params['v_FY1'],
        smoke_deploy_time=best_params['t_deploy'],
        smoke_explode_delay=best_params['t_fuse']
    )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    return best_params


def compare_with_pso(ga_params: Dict[str, float], ga_fitness: float):
    """与PSO结果进行比较"""
    print("="*60)
    print("GA与PSO结果对比分析")
    print("="*60)
    
    # 这里可以手动输入PSO的结果进行比较
    # 或者从文件读取PSO结果
    print("请手动输入PSO的最优结果进行比较：")
    
    try:
        pso_v_FY1 = float(input("PSO - 无人机速度 (m/s): "))
        pso_theta_FY1 = float(input("PSO - 无人机方向 (度): "))
        pso_t_deploy = float(input("PSO - 投放时间 (s): "))
        pso_t_fuse = float(input("PSO - 引信延时 (s): "))
        pso_fitness = float(input("PSO - 最大遮蔽时长 (s): "))
        
        print(f"\n对比结果：")
        print(f"{'参数':<15} {'GA结果':<12} {'PSO结果':<12} {'差异':<12}")
        print("-" * 55)
        print(f"{'速度 (m/s)':<15} {ga_params['v_FY1']:<12.2f} {pso_v_FY1:<12.2f} {abs(ga_params['v_FY1']-pso_v_FY1):<12.2f}")
        print(f"{'方向 (度)':<15} {ga_params['theta_FY1']:<12.2f} {pso_theta_FY1:<12.2f} {abs(ga_params['theta_FY1']-pso_theta_FY1):<12.2f}")
        print(f"{'投放时间 (s)':<15} {ga_params['t_deploy']:<12.3f} {pso_t_deploy:<12.3f} {abs(ga_params['t_deploy']-pso_t_deploy):<12.3f}")
        print(f"{'引信延时 (s)':<15} {ga_params['t_fuse']:<12.3f} {pso_t_fuse:<12.3f} {abs(ga_params['t_fuse']-pso_t_fuse):<12.3f}")
        print(f"{'遮蔽时长 (s)':<15} {ga_fitness:<12.6f} {pso_fitness:<12.6f} {abs(ga_fitness-pso_fitness):<12.6f}")
        
        # 分析一致性
        fitness_diff = abs(ga_fitness - pso_fitness)
        if fitness_diff < 0.001:
            print(f"\n✅ 结果高度一致！两种算法找到了相似的最优解。")
        elif fitness_diff < 0.01:
            print(f"\n⚠️  结果基本一致，存在小幅差异，这是正常的。")
        else:
            print(f"\n❌ 结果存在较大差异，建议检查算法参数或增加迭代次数。")
            
    except (ValueError, KeyboardInterrupt):
        print("跳过PSO结果比较")


def main():
    """主函数"""
    print("问题2：使用遗传算法求解最优无人机策略（交叉验证PSO结果）")
    
    # 设置GA参数 - 优化参数设置
    ga_params = {
        'population_size': 80,        # 增加种群大小
        'max_generations': 150,       # 增加代数
        'crossover_rate': 0.9,        # 提高交叉率
        'mutation_rate': 0.15,        # 提高变异率
        'elite_size': 5,              # 增加精英个体数
        'tournament_size': 5,         # 增加锦标赛规模
        'use_parallel': True
    }
    
    print(f"\n遗传算法参数：")
    for key, value in ga_params.items():
        print(f"  {key}: {value}")
    
    # 创建优化器
    optimizer = GA_Optimizer(**ga_params)
    
    print(f"\n搜索空间边界：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    best_individual, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    
    # 分析结果
    best_params = analyze_ga_results(best_individual, best_fitness, optimizer.bounds)
    
    # 绘制收敛曲线
    optimizer.plot_convergence()
    
    # 与PSO结果比较
    compare_with_pso(best_params, best_fitness)
    
    # 保存结果
    results = {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'ga_params': ga_params,
        'bounds': optimizer.bounds,
        'best_fitness_history': optimizer.best_fitness_history,
        'avg_fitness_history': optimizer.avg_fitness_history
    }
    
    print(f"\n遗传算法优化结果已保存")
    return results


if __name__ == "__main__":
    results = main()
