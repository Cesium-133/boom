#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUMCM2025 Problem 4 - 高性能服务器版本 (80核心并行优化)
多无人机协同烟幕遮蔽优化 - 差分进化算法

服务器配置要求：
- CPU: 80+ 核心
- RAM: 32GB+
- Python 3.8+

性能目标：
- 80个个体完全并行计算
- 每代评估时间 < 100ms
- 支持长时间运行和检查点保存
"""

import numpy as np
import time
import random
import os
import sys
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import signal
import logging
from pathlib import Path
import psutil

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solver.core import (
    calculate_multi_uav_single_smoke_masking,
    calculate_multi_uav_single_smoke_masking_multiple,
    clear_multiple_cache
)
from solver.config import BOUNDS

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    """服务器配置"""
    max_cores: int = 80              # 最大核心数
    memory_limit_gb: int = 32        # 内存限制(GB)
    checkpoint_interval: int = 50    # 检查点保存间隔
    checkpoint_dir: str = "./checkpoints"
    log_level: str = "INFO"
    process_timeout: int = 300       # 进程超时时间(秒)

class ServerOptimizedDE:
    """针对80核心服务器优化的差分进化算法"""
    
    def __init__(self,
                 population_size: int = 80,        # 与核心数匹配
                 max_generations: int = 2000,
                 bounds: Dict[str, Tuple[float, float]] = None,
                 server_config: ServerConfig = None,
                 masking_mode: str = "independent",
                 F: float = 0.5,
                 CR: float = 0.9,
                 mutation_strategy: str = "rand/1",
                 use_adaptive: bool = True,
                 checkpoint_name: str = "problem4_server"):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.bounds = bounds or BOUNDS
        self.server_config = server_config or ServerConfig()
        self.masking_mode = masking_mode
        self.F = F
        self.CR = CR
        self.mutation_strategy = mutation_strategy
        self.use_adaptive = use_adaptive
        self.checkpoint_name = checkpoint_name
        
        # 服务器资源检查
        self._check_server_resources()
        
        # 初始化并行配置
        self._setup_parallel_config()
        
        # 创建检查点目录
        Path(self.server_config.checkpoint_dir).mkdir(exist_ok=True)
        
        # 性能统计
        self.performance_stats = {
            'generation_times': [],
            'evaluation_times': [],
            'parallel_efficiency': [],
            'memory_usage': []
        }
        
        logger.info(f"服务器版DE初始化完成 - {self.n_processes}核心并行")

    def _check_server_resources(self):
        """检查服务器资源"""
        cpu_count = mp.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"服务器资源检测:")
        logger.info(f"  CPU核心数: {cpu_count}")
        logger.info(f"  内存容量: {memory_gb:.1f}GB")
        
        if cpu_count < 40:
            logger.warning(f"CPU核心数({cpu_count})低于推荐值(40+)")
        if memory_gb < 16:
            logger.warning(f"内存容量({memory_gb:.1f}GB)低于推荐值(16GB+)")
            
        # 从环境变量读取配置（如果有）
        max_cores_env = os.getenv('MAX_CORES')
        if max_cores_env:
            try:
                self.server_config.max_cores = int(max_cores_env)
                logger.info(f"从环境变量设置最大核心数: {self.server_config.max_cores}")
            except ValueError:
                logger.warning(f"无效的MAX_CORES环境变量: {max_cores_env}")
        
        max_memory_env = os.getenv('MAX_MEMORY_GB')
        if max_memory_env:
            try:
                self.server_config.memory_limit_gb = int(max_memory_env)
                logger.info(f"从环境变量设置内存限制: {self.server_config.memory_limit_gb}GB")
            except ValueError:
                logger.warning(f"无效的MAX_MEMORY_GB环境变量: {max_memory_env}")
            
        # 调整配置
        self.server_config.max_cores = min(self.server_config.max_cores, cpu_count)

    def _setup_parallel_config(self):
        """设置并行配置"""
        # 使用所有可用核心，但不超过population_size
        self.n_processes = min(
            self.server_config.max_cores,
            self.population_size,
            mp.cpu_count()
        )
        
        # 进程池配置
        self.process_pool_config = {
            'max_workers': self.n_processes,
            'mp_context': mp.get_context('spawn'),  # 使用spawn避免fork问题
        }
        
        logger.info(f"并行配置: {self.n_processes}个进程，每个体独立进程")

    def _create_bounds_list(self):
        """创建边界列表"""
        bounds_list = []
        param_names = [
            'uav_a_direction', 'uav_a_speed',
            'uav_b_direction', 'uav_b_speed', 
            'uav_c_direction', 'uav_c_speed',
            'smoke_a_deploy_time', 'smoke_a_explode_delay',
            'smoke_b_deploy_time', 'smoke_b_explode_delay',
            'smoke_c_deploy_time', 'smoke_c_explode_delay'
        ]
        
        for param in param_names:
            if param in self.bounds:
                bounds_list.append(self.bounds[param])
            else:
                raise KeyError(f"参数 {param} 未在边界中定义")
        
        return bounds_list

    def save_checkpoint(self, generation: int, population: np.ndarray, 
                       fitness: np.ndarray, best_fitness: float, best_individual: np.ndarray):
        """保存检查点"""
        checkpoint_data = {
            'generation': generation,
            'population': population,
            'fitness': fitness,
            'best_fitness': best_fitness,
            'best_individual': best_individual,
            'performance_stats': self.performance_stats,
            'config': {
                'population_size': self.population_size,
                'max_generations': self.max_generations,
                'bounds': self.bounds,
                'masking_mode': self.masking_mode,
                'F': self.F,
                'CR': self.CR
            }
        }
        
        checkpoint_file = Path(self.server_config.checkpoint_dir) / f"{self.checkpoint_name}_gen_{generation}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        logger.info(f"检查点已保存: {checkpoint_file}")

    def load_checkpoint(self, checkpoint_file: str) -> Optional[Dict]:
        """加载检查点"""
        try:
            with open(checkpoint_file, 'rb') as f:
                checkpoint_data = pickle.load(f)
            logger.info(f"检查点已加载: {checkpoint_file}")
            return checkpoint_data
        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return None

    def optimize(self, resume_from: Optional[str] = None):
        """服务器优化主循环"""
        logger.info("="*80)
        logger.info("🚀 启动服务器版差分进化优化")
        logger.info(f"种群规模: {self.population_size}, 最大代数: {self.max_generations}")
        logger.info(f"并行核心: {self.n_processes}, 遮蔽模式: {self.masking_mode}")
        logger.info("="*80)
        
        # 检查是否从检查点恢复
        start_generation = 0
        if resume_from:
            checkpoint_data = self.load_checkpoint(resume_from)
            if checkpoint_data:
                population = checkpoint_data['population']
                fitness = checkpoint_data['fitness']
                best_fitness = checkpoint_data['best_fitness']
                best_individual = checkpoint_data['best_individual']
                start_generation = checkpoint_data['generation'] + 1
                self.performance_stats = checkpoint_data.get('performance_stats', self.performance_stats)
                logger.info(f"从第{start_generation}代恢复优化")
            else:
                logger.error("检查点加载失败，从头开始")
                resume_from = None

        # 初始化种群
        if not resume_from:
            bounds_list = self._create_bounds_list()
            population = self._initialize_population(bounds_list)
            fitness = self._evaluate_population_server(population, bounds_list)
            best_idx = np.argmax(fitness)
            best_fitness = fitness[best_idx]
            best_individual = population[best_idx].copy()
            logger.info(f"初始种群最优适应度: {best_fitness:.6f}")

        # 设置信号处理（优雅退出）
        self._setup_signal_handlers()
        
        try:
            # 主优化循环
            for generation in range(start_generation, self.max_generations):
                generation_start = time.time()
                
                # 生成试验向量
                trials = self._generate_trials_server(population, bounds_list)
                
                # 服务器级并行评估
                evaluation_start = time.time()
                trial_fitness = self._evaluate_population_server(trials, bounds_list)
                evaluation_time = time.time() - evaluation_start
                
                # 选择操作
                for i in range(self.population_size):
                    if trial_fitness[i] > fitness[i]:
                        population[i] = trials[i]
                        fitness[i] = trial_fitness[i]
                        
                        if trial_fitness[i] > best_fitness:
                            best_fitness = trial_fitness[i]
                            best_individual = trials[i].copy()
                
                generation_time = time.time() - generation_start
                
                # 性能统计
                self._update_performance_stats(generation_time, evaluation_time)
                
                # 输出进度
                if generation % 10 == 0 or generation < 10:
                    self._print_progress(generation, best_fitness, generation_time, evaluation_time)
                
                # 保存检查点
                if generation % self.server_config.checkpoint_interval == 0:
                    self._save_checkpoint_async(generation, population, fitness, best_fitness, best_individual)
                
                # 自适应参数调整
                if self.use_adaptive:
                    self._adaptive_parameter_update(generation)

        except KeyboardInterrupt:
            logger.info("收到中断信号，正在优雅退出...")
            self._save_final_results(generation, best_fitness, best_individual)
        
        # 最终结果
        logger.info("="*80)
        logger.info("🎯 优化完成!")
        logger.info(f"最优适应度: {best_fitness:.6f}")
        logger.info(f"最优参数: {best_individual}")
        self._print_performance_summary()
        
        return {
            'best_fitness': best_fitness,
            'best_individual': best_individual,
            'performance_stats': self.performance_stats
        }

    def _initialize_population(self, bounds_list: List[Tuple[float, float]]) -> np.ndarray:
        """初始化种群"""
        population = np.random.rand(self.population_size, len(bounds_list))
        for i, (lower, upper) in enumerate(bounds_list):
            population[:, i] = population[:, i] * (upper - lower) + lower
        return population

    def _generate_trials_server(self, population: np.ndarray, bounds_list: List) -> np.ndarray:
        """生成试验向量（服务器优化版）"""
        trials = np.zeros_like(population)
        
        for i in range(self.population_size):
            # 选择三个不同的个体
            candidates = list(range(self.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # DE/rand/1 变异
            mutant = population[r1] + self.F * (population[r2] - population[r3])
            
            # 边界处理
            for j, (lower, upper) in enumerate(bounds_list):
                if mutant[j] < lower:
                    mutant[j] = lower + np.random.rand() * (upper - lower) * 0.1
                elif mutant[j] > upper:
                    mutant[j] = upper - np.random.rand() * (upper - lower) * 0.1
            
            # 交叉
            trial = population[i].copy()
            crossover_mask = np.random.rand(len(bounds_list)) < self.CR
            crossover_mask[np.random.randint(len(bounds_list))] = True  # 确保至少一个基因交叉
            trial[crossover_mask] = mutant[crossover_mask]
            
            trials[i] = trial
        
        return trials

    def _evaluate_population_server(self, population: np.ndarray, bounds_list: List) -> np.ndarray:
        """服务器级并行种群评估（每个体一个进程）"""
        fitness = np.zeros(self.population_size)
        
        # 准备每个个体的数据
        individual_data = [(population[i], bounds_list, i) for i in range(self.population_size)]
        
        # 使用所有可用核心并行评估
        with ProcessPoolExecutor(**self.process_pool_config) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(evaluate_individual_server, data): data[2] 
                for data in individual_data
            }
            
            # 收集结果
            completed = 0
            for future in as_completed(future_to_idx, timeout=self.server_config.process_timeout):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    fitness[idx] = result
                    completed += 1
                    
                    # 进度显示（每20个个体）
                    if completed % 20 == 0:
                        print(f"    已完成 {completed}/{self.population_size} 个体评估", end='\r')
                        
                except Exception as e:
                    logger.error(f"个体{idx}评估失败: {e}")
                    fitness[idx] = -1000.0
        
        return fitness

    def _update_performance_stats(self, generation_time: float, evaluation_time: float):
        """更新性能统计"""
        self.performance_stats['generation_times'].append(generation_time)
        self.performance_stats['evaluation_times'].append(evaluation_time)
        
        # 并行效率 = 实际加速比 / 理论加速比
        theoretical_time = evaluation_time * self.n_processes  # 串行时间估计
        actual_speedup = theoretical_time / evaluation_time if evaluation_time > 0 else 0
        parallel_efficiency = actual_speedup / self.n_processes if self.n_processes > 0 else 0
        self.performance_stats['parallel_efficiency'].append(parallel_efficiency)
        
        # 内存使用
        memory_usage = psutil.virtual_memory().percent
        self.performance_stats['memory_usage'].append(memory_usage)

    def _print_progress(self, generation: int, best_fitness: float, 
                       generation_time: float, evaluation_time: float):
        """打印进度信息"""
        avg_per_individual = evaluation_time / self.population_size * 1000  # ms
        parallel_efficiency = self.performance_stats['parallel_efficiency'][-1] * 100
        memory_usage = self.performance_stats['memory_usage'][-1]
        
        print(f"\n第{generation:4d}代 | 最优: {best_fitness:8.4f} | "
              f"总时间: {generation_time:6.2f}s | 评估: {evaluation_time:6.2f}s")
        print(f"         | 每个体: {avg_per_individual:5.1f}ms | "
              f"并行效率: {parallel_efficiency:5.1f}% | 内存: {memory_usage:4.1f}%")

    def _print_performance_summary(self):
        """打印性能总结"""
        if not self.performance_stats['generation_times']:
            return
            
        avg_gen_time = np.mean(self.performance_stats['generation_times'])
        avg_eval_time = np.mean(self.performance_stats['evaluation_times'])
        avg_efficiency = np.mean(self.performance_stats['parallel_efficiency']) * 100
        avg_memory = np.mean(self.performance_stats['memory_usage'])
        
        print("\n" + "="*80)
        print("📊 性能总结:")
        print(f"  平均每代时间: {avg_gen_time:.2f}s")
        print(f"  平均评估时间: {avg_eval_time:.2f}s")
        print(f"  平均每个体时间: {avg_eval_time/self.population_size*1000:.1f}ms")
        print(f"  平均并行效率: {avg_efficiency:.1f}%")
        print(f"  平均内存使用: {avg_memory:.1f}%")
        print(f"  使用核心数: {self.n_processes}")
        print("="*80)

    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"收到信号 {signum}，准备优雅退出...")
            raise KeyboardInterrupt()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _save_checkpoint_async(self, generation: int, population: np.ndarray, 
                              fitness: np.ndarray, best_fitness: float, best_individual: np.ndarray):
        """异步保存检查点"""
        try:
            self.save_checkpoint(generation, population, fitness, best_fitness, best_individual)
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")

    def _save_final_results(self, generation: int, best_fitness: float, best_individual: np.ndarray):
        """保存最终结果"""
        results_file = Path(self.server_config.checkpoint_dir) / f"{self.checkpoint_name}_final_results.txt"
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write(f"最终优化结果 (第{generation}代中断)\n")
            f.write(f"最优适应度: {best_fitness:.6f}\n")
            f.write(f"最优参数: {best_individual}\n")
            f.write(f"性能统计: {self.performance_stats}\n")
        
        logger.info(f"最终结果已保存: {results_file}")

    def _adaptive_parameter_update(self, generation: int):
        """自适应参数更新"""
        # 简单的自适应策略
        progress = generation / self.max_generations
        self.F = 0.5 + 0.3 * (1 - progress)  # F从0.8递减到0.5
        self.CR = 0.9 - 0.2 * progress       # CR从0.9递减到0.7


def evaluate_individual_server(data: Tuple) -> float:
    """服务器版个体评估函数（独立进程）"""
    individual, bounds_list, idx = data
    
    try:
        # 解码参数
        params = {
            'uav_a_direction': individual[0], 'uav_a_speed': individual[1],
            'uav_b_direction': individual[2], 'uav_b_speed': individual[3],
            'uav_c_direction': individual[4], 'uav_c_speed': individual[5],
            'smoke_a_deploy_time': individual[6], 'smoke_a_explode_delay': individual[7],
            'smoke_b_deploy_time': individual[8], 'smoke_b_explode_delay': individual[9],
            'smoke_c_deploy_time': individual[10], 'smoke_c_explode_delay': individual[11]
        }
        
        # 计算适应度
        duration = calculate_multi_uav_single_smoke_masking(**params)
        return duration
        
    except Exception as e:
        # 记录错误但不中断整个优化过程
        return -1000.0


def main():
    """主函数"""
    print("🖥️  CUMCM2025 Problem 4 - 服务器版 (80核心优化)")
    print("="*80)
    
    # 服务器配置
    server_config = ServerConfig(
        max_cores=80,
        memory_limit_gb=32,
        checkpoint_interval=50,
        checkpoint_dir="./server_checkpoints"
    )
    
    # 创建优化器
    optimizer = ServerOptimizedDE(
        population_size=80,      # 与核心数匹配
        max_generations=2000,    # 长时间运行
        server_config=server_config,
        masking_mode="independent",
        use_adaptive=True,
        checkpoint_name="problem4_server_80core"
    )
    
    # 检查是否有检查点可以恢复
    checkpoint_dir = Path(server_config.checkpoint_dir)
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("problem4_server_80core_gen_*.pkl"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            print(f"发现检查点: {latest_checkpoint}")
            resume = input("是否从检查点恢复？(y/n): ").lower().strip()
            if resume == 'y':
                result = optimizer.optimize(resume_from=str(latest_checkpoint))
            else:
                result = optimizer.optimize()
        else:
            result = optimizer.optimize()
    else:
        result = optimizer.optimize()
    
    print(f"\n🎯 服务器优化完成!")
    print(f"最优适应度: {result['best_fitness']:.6f}")
    print(f"最优参数: {result['best_individual']}")


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 