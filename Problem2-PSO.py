"""
问题2：使用粒子群优化算法求解最优无人机策略

目标：找到最优的无人机速度、飞行方向、烟幕弹投放时间和引信延时，
使得有效遮蔽时长最大化。

决策变量：
- v_FY1: 无人机速度 [70, 140] m/s
- θ_FY1: 无人机飞行方向 [0, 360] 度
- t_deploy: 烟幕弹投放时间 [0, T_max] s
- t_fuse: 烟幕弹引信相对时间 [0, T_fuse_max] s
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 导入求解器
from solver import calculate_single_uav_single_smoke_masking, TARGETS, MISSILES
from solver.trajectory import TrajectoryCalculator

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# 全局函数，用于并行计算
def evaluate_position_fitness(position_data):
    """评估位置适应度的全局函数"""
    position, bounds_keys = position_data
    
    try:
        # 解码位置
        params = {bounds_keys[i]: position[i] for i in range(len(bounds_keys))}
        
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


class PSO_Optimizer:
    """粒子群优化器"""
    
    def __init__(self, 
                 n_particles: int = 30,
                 max_iterations: int = 100,
                 w_start: float = 0.9,
                 w_end: float = 0.4,
                 c1: float = 2.0,
                 c2: float = 2.0,
                 bounds: Dict[str, Tuple[float, float]] = None):
        """
        初始化PSO优化器
        
        Args:
            n_particles: 粒子数量
            max_iterations: 最大迭代次数
            w_start: 初始惯性权重
            w_end: 终止惯性权重
            c1: 个体学习因子
            c2: 全局学习因子
            bounds: 变量边界字典
        """
        self.n_particles = n_particles
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        
        # 设置默认边界
        if bounds is None:
            self.bounds = self._calculate_default_bounds()
        else:
            self.bounds = bounds
            
        # 维度数量
        self.n_dims = len(self.bounds)
        
        # 初始化粒子群
        self.particles_pos = np.zeros((n_particles, self.n_dims))
        self.particles_vel = np.zeros((n_particles, self.n_dims))
        self.particles_fitness = np.zeros(n_particles)
        
        # 个体最优
        self.pbest_pos = np.zeros((n_particles, self.n_dims))
        self.pbest_fitness = np.zeros(n_particles)
        
        # 全局最优
        self.gbest_pos = np.zeros(self.n_dims)
        self.gbest_fitness = -np.inf
        
        # 记录历史
        self.fitness_history = []
        
        # 并行计算设置
        self.use_parallel = True
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), n_particles)
            print(f"将使用 {self.n_processes} 个进程进行并行计算")
        
    def _calculate_default_bounds(self) -> Dict[str, Tuple[float, float]]:
        """计算默认边界"""
        print("正在计算搜索空间边界...")
        
        # 计算导弹到达虚假目标的时间
        traj_calc = TrajectoryCalculator()
        missile_traj = traj_calc.create_missile_trajectory("M1")
        
        # 计算导弹到达原点的时间
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
        
        # 设置边界
        bounds = {
            'v_FY1': (70.0, 140.0),      # 无人机速度
            'theta_FY1': (0.0, 360.0),   # 无人机方向
            't_deploy': (0.1, t_max - 1.0),  # 投放时间，留1秒余量
            't_fuse': (1.42, t_max - 1.0)        # 引信延时，1.42保证不会炸到无人机
        }
        
        return bounds
    
    def _decode_position(self, position: np.ndarray) -> Dict[str, float]:
        """将位置向量解码为参数字典"""
        keys = list(self.bounds.keys())
        return {keys[i]: position[i] for i in range(len(keys))}
    
    def _objective_function(self, position: np.ndarray) -> float:
        """目标函数：计算有效遮蔽时长"""
        try:
            params = self._decode_position(position)
            
            # 调用问题1的计算函数
            duration = calculate_single_uav_single_smoke_masking(
                uav_direction=params['theta_FY1'],
                uav_speed=params['v_FY1'],
                smoke_deploy_time=params['t_deploy'],
                smoke_explode_delay=params['t_fuse']
            )
            
            return duration
            
        except Exception as e:
            # 如果计算出错，返回很小的值
            print(f"计算错误: {e}")
            return -1000.0
    
    def _clip_to_bounds(self, position: np.ndarray) -> np.ndarray:
        """将位置限制在边界内"""
        keys = list(self.bounds.keys())
        for i, key in enumerate(keys):
            min_val, max_val = self.bounds[key]
            position[i] = np.clip(position[i], min_val, max_val)
        return position
    
    def _initialize_particles(self):
        """初始化粒子群"""
        print("初始化粒子群...")
        
        keys = list(self.bounds.keys())
        
        for i in range(self.n_particles):
            # 随机初始化位置
            for j, key in enumerate(keys):
                min_val, max_val = self.bounds[key]
                self.particles_pos[i, j] = np.random.uniform(min_val, max_val)
            
            # 随机初始化速度（较小的初始速度）
            for j, key in enumerate(keys):
                min_val, max_val = self.bounds[key]
                range_val = max_val - min_val
                self.particles_vel[i, j] = np.random.uniform(-range_val*0.1, range_val*0.1)
        
        # 并行计算初始适应度
        if self.use_parallel:
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                tasks = [(self.particles_pos[i], keys) for i in range(self.n_particles)]
                futures = {executor.submit(evaluate_position_fitness, task): i for i, task in enumerate(tasks)}
                
                for future in as_completed(futures):
                    i = futures[future]
                    fitness_value = future.result()
                    self.particles_fitness[i] = fitness_value
                    
                    # 设置个体最优
                    self.pbest_pos[i] = self.particles_pos[i].copy()
                    self.pbest_fitness[i] = fitness_value
                    
                    # 更新全局最优
                    if fitness_value > self.gbest_fitness:
                        self.gbest_fitness = fitness_value
                        self.gbest_pos = self.particles_pos[i].copy()
        else:
            for i in range(self.n_particles):
                # 计算初始适应度
                self.particles_fitness[i] = self._objective_function(self.particles_pos[i])
                
                # 设置个体最优
                self.pbest_pos[i] = self.particles_pos[i].copy()
                self.pbest_fitness[i] = self.particles_fitness[i]
                
                # 更新全局最优
                if self.particles_fitness[i] > self.gbest_fitness:
                    self.gbest_fitness = self.particles_fitness[i]
                    self.gbest_pos = self.particles_pos[i].copy()
                
                print(f"粒子 {i+1}/{self.n_particles} 初始化完成，适应度: {self.particles_fitness[i]:.4f}")
        
        print(f"初始化完成，全局最优适应度: {self.gbest_fitness:.4f}")
    
    def _update_inertia_weight(self, iteration: int) -> float:
        """线性递减惯性权重"""
        return self.w_start - (self.w_start - self.w_end) * iteration / self.max_iterations
    
    def optimize(self) -> Tuple[np.ndarray, float]:
        """执行PSO优化"""
        print("="*60)
        print("开始粒子群优化算法")
        print("="*60)
        
        # 初始化粒子群
        self._initialize_particles()
        
        # 主优化循环
        for iteration in range(self.max_iterations):
            print(f"\n第 {iteration+1}/{self.max_iterations} 次迭代")
            
            # 更新惯性权重
            w = self._update_inertia_weight(iteration)
            
            # 更新每个粒子
            for i in range(self.n_particles):
                # 生成随机数
                r1 = np.random.random(self.n_dims)
                r2 = np.random.random(self.n_dims)
                
                # 更新速度
                self.particles_vel[i] = (
                    w * self.particles_vel[i] +
                    self.c1 * r1 * (self.pbest_pos[i] - self.particles_pos[i]) +
                    self.c2 * r2 * (self.gbest_pos - self.particles_pos[i])
                )
                
                # 更新位置
                self.particles_pos[i] = self.particles_pos[i] + self.particles_vel[i]
                
                # 边界处理
                self.particles_pos[i] = self._clip_to_bounds(self.particles_pos[i])
                
            # 并行计算新适应度
            if self.use_parallel:
                with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                    keys = list(self.bounds.keys())
                    tasks = [(self.particles_pos[i], keys) for i in range(self.n_particles)]
                    futures = {executor.submit(evaluate_position_fitness, task): i for i, task in enumerate(tasks)}
                    
                    for future in as_completed(futures):
                        i = futures[future]
                        new_fitness = future.result()
                        self.particles_fitness[i] = new_fitness
                        
                        # 更新个体最优
                        if new_fitness > self.pbest_fitness[i]:
                            self.pbest_fitness[i] = new_fitness
                            self.pbest_pos[i] = self.particles_pos[i].copy()
                        
                        # 更新全局最优
                        if new_fitness > self.gbest_fitness:
                            self.gbest_fitness = new_fitness
                            self.gbest_pos = self.particles_pos[i].copy()
                            print(f"  发现新的全局最优! 适应度: {self.gbest_fitness:.6f}")
            else:
                for i in range(self.n_particles):
                    # 计算新适应度
                    new_fitness = self._objective_function(self.particles_pos[i])
                    self.particles_fitness[i] = new_fitness
                    
                    # 更新个体最优
                    if new_fitness > self.pbest_fitness[i]:
                        self.pbest_fitness[i] = new_fitness
                        self.pbest_pos[i] = self.particles_pos[i].copy()
                    
                    # 更新全局最优
                    if new_fitness > self.gbest_fitness:
                        self.gbest_fitness = new_fitness
                        self.gbest_pos = self.particles_pos[i].copy()
                        print(f"  发现新的全局最优! 适应度: {self.gbest_fitness:.6f}")
            
            # 记录历史
            self.fitness_history.append(self.gbest_fitness)
            
            # 输出迭代信息
            avg_fitness = np.mean(self.particles_fitness)
            print(f"  惯性权重: {w:.3f}")
            print(f"  当前全局最优: {self.gbest_fitness:.6f}")
            print(f"  当前平均适应度: {avg_fitness:.6f}")
            
            # 简单的收敛判断
            if iteration > 10:
                recent_improvement = max(self.fitness_history[-10:]) - min(self.fitness_history[-10:])
                if recent_improvement < 1e-6:
                    print(f"  算法收敛，提前结束于第 {iteration+1} 次迭代")
                    break
        
        return self.gbest_pos, self.gbest_fitness
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, 'b-', linewidth=2)
        plt.title('PSO算法收敛曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('全局最优适应度（有效遮蔽时长）')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def analyze_results(best_position: np.ndarray, best_fitness: float, bounds: Dict[str, Tuple[float, float]]):
    """分析和展示结果"""
    print("="*60)
    print("优化结果分析")
    print("="*60)
    
    # 解码最优解
    keys = list(bounds.keys())
    best_params = {keys[i]: best_position[i] for i in range(len(keys))}
    
    print(f"\n最优策略参数：")
    print(f"  无人机速度 (v_FY1): {best_params['v_FY1']:.2f} m/s")
    print(f"  无人机方向 (θ_FY1): {best_params['theta_FY1']:.2f}°")
    print(f"  投放时间 (t_deploy): {best_params['t_deploy']:.3f} s")
    print(f"  引信延时 (t_fuse): {best_params['t_fuse']:.3f} s")
    print(f"  起爆时间: {best_params['t_deploy'] + best_params['t_fuse']:.3f} s")
    
    print(f"\n最大有效遮蔽时长: {best_fitness:.6f} 秒")
    
    # 与理论值比较
    theoretical_optimal = 4.8
    gap = abs(best_fitness - theoretical_optimal)
    gap_percentage = (gap / theoretical_optimal) * 100
    
    print(f"\n与理论最优值比较：")
    print(f"  理论最优值: ≈{theoretical_optimal:.1f} 秒")
    print(f"  找到的最优值: {best_fitness:.6f} 秒")
    print(f"  差距: {gap:.6f} 秒 ({gap_percentage:.2f}%)")
    
    if gap_percentage < 1.0:
        print(f"  🎯 优秀！非常接近理论最优值")
    elif gap_percentage < 5.0:
        print(f"  ✅ 良好！较为接近理论最优值")
    else:
        print(f"  ⚠️  有改进空间，可尝试增强版PSO算法")
    
    # 验证结果
    print(f"\n验证计算...")
    verification_result = calculate_single_uav_single_smoke_masking(
        uav_direction=best_params['theta_FY1'],
        uav_speed=best_params['v_FY1'],
        smoke_deploy_time=best_params['t_deploy'],
        smoke_explode_delay=best_params['t_fuse']
    )
    print(f"验证结果: {verification_result:.6f} 秒")
    
    # 分析参数合理性
    print(f"\n参数合理性分析：")
    
    # 无人机速度
    speed_ratio = (best_params['v_FY1'] - bounds['v_FY1'][0]) / (bounds['v_FY1'][1] - bounds['v_FY1'][0])
    if speed_ratio < 0.3:
        print(f"  无人机速度偏低，可能优先考虑精确定位")
    elif speed_ratio > 0.7:
        print(f"  无人机速度偏高，可能优先考虑快速到达")
    else:
        print(f"  无人机速度适中，平衡了速度和精度")
    
    # 飞行方向
    if 315 <= best_params['theta_FY1'] or best_params['theta_FY1'] <= 45:
        print(f"  无人机主要向东飞行")
    elif 45 < best_params['theta_FY1'] <= 135:
        print(f"  无人机主要向北飞行")
    elif 135 < best_params['theta_FY1'] <= 225:
        print(f"  无人机主要向西飞行")
    else:
        print(f"  无人机主要向南飞行")
    
    # 时间参数
    total_time = best_params['t_deploy'] + best_params['t_fuse']
    if best_params['t_deploy'] < 1:
        print(f"  投放时间很早，可能为了获得更长的遮蔽时间")
    if best_params['t_fuse'] > 5:
        print(f"  引信延时较长，可能为了等待最佳遮蔽时机")
    
    return best_params


def sensitivity_analysis(best_params: Dict[str, float]):
    """敏感性分析"""
    print(f"\n执行敏感性分析...")
    
    base_fitness = calculate_single_uav_single_smoke_masking(
        uav_direction=best_params['theta_FY1'],
        uav_speed=best_params['v_FY1'],
        smoke_deploy_time=best_params['t_deploy'],
        smoke_explode_delay=best_params['t_fuse']
    )
    
    # 对每个参数进行敏感性分析
    sensitivity = {}
    perturbation = 0.05  # 5%的扰动
    
    for param_name, param_value in best_params.items():
        if param_name == 'v_FY1':
            delta = param_value * perturbation
            fitness_up = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=param_value + delta,
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=best_params['t_fuse']
            )
            fitness_down = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=param_value - delta,
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=best_params['t_fuse']
            )
        elif param_name == 'theta_FY1':
            delta = 5.0  # 5度
            fitness_up = calculate_single_uav_single_smoke_masking(
                uav_direction=param_value + delta,
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=best_params['t_fuse']
            )
            fitness_down = calculate_single_uav_single_smoke_masking(
                uav_direction=param_value - delta,
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=best_params['t_fuse']
            )
        elif param_name == 't_deploy':
            delta = param_value * perturbation
            fitness_up = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=param_value + delta,
                smoke_explode_delay=best_params['t_fuse']
            )
            fitness_down = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=param_value - delta,
                smoke_explode_delay=best_params['t_fuse']
            )
        elif param_name == 't_fuse':
            delta = param_value * perturbation
            fitness_up = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=param_value + delta
            )
            fitness_down = calculate_single_uav_single_smoke_masking(
                uav_direction=best_params['theta_FY1'],
                uav_speed=best_params['v_FY1'],
                smoke_deploy_time=best_params['t_deploy'],
                smoke_explode_delay=param_value - delta
            )
        
        # 计算敏感性
        sensitivity[param_name] = abs(fitness_up - fitness_down) / (2 * delta)
    
    print("参数敏感性分析结果：")
    for param, sens in sorted(sensitivity.items(), key=lambda x: x[1], reverse=True):
        print(f"  {param}: {sens:.6f}")


def main():
    """主函数"""
    print("问题2：使用粒子群优化算法求解最优无人机策略")
    
    # 设置PSO参数 - 优化设置以接近理论最优值4.8
    pso_params = {
        'n_particles': 50,        # 增加粒子数量
        'max_iterations': 150,    # 增加迭代次数
        'w_start': 0.9,
        'w_end': 0.1,            # 降低最终惯性权重
        'c1': 2.5,               # 增加个体学习因子
        'c2': 1.5                # 调整全局学习因子
    }
    
    print(f"\nPSO算法参数：")
    for key, value in pso_params.items():
        print(f"  {key}: {value}")
    
    # 创建优化器
    optimizer = PSO_Optimizer(**pso_params)
    
    print(f"\n搜索空间边界：")
    for param, (min_val, max_val) in optimizer.bounds.items():
        print(f"  {param}: [{min_val:.2f}, {max_val:.2f}]")
    
    # 执行优化
    start_time = time.time()
    best_position, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    
    # 分析结果
    best_params = analyze_results(best_position, best_fitness, optimizer.bounds)
    
    # 敏感性分析
    sensitivity_analysis(best_params)
    
    # 绘制收敛曲线
    optimizer.plot_convergence()
    
    # 保存结果
    results = {
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'pso_params': pso_params,
        'bounds': optimizer.bounds,
        'fitness_history': optimizer.fitness_history
    }
    
    print(f"\n优化结果已保存")
    return results


if __name__ == "__main__":
    results = main()
