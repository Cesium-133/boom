#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUMCM2025 Problem 5 - 多无人机多导弹协同烟幕遮蔽优化
5架无人机，每架最多3枚烟幕弹，对抗3枚来袭导弹（M1、M2、M3）

优化目标：最大化对所有导弹的综合遮蔽时长
"""

import numpy as np
import time
import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
import logging
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solver.core import calculate_problem5_multi_uav_multi_missile_masking
from solver.config import calculate_problem5_bounds

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OptimizationConfig:
    """优化配置"""
    population_size: int = 100        # 种群规模（5*30=150维，需要大种群）
    max_generations: int = 1000       # 最大代数
    F: float = 0.5                    # 差分权重
    CR: float = 0.9                   # 交叉概率
    use_parallel: bool = True         # 是否并行
    checkpoint_interval: int = 50     # 检查点保存间隔


def evaluate_individual_problem5(individual_data: Tuple) -> float:
    """
    评估单个个体的适应度函数
    
    Args:
        individual_data: (individual, bounds_list) 元组
        
    Returns:
        适应度值（综合遮蔽时长）
    """
    individual, bounds_list = individual_data
    
    try:
        # 解码参数 - 按照边界定义的顺序
        param_names = [
            # 5架无人机的方向和速度
            'uav_1_direction', 'uav_1_speed',
            'uav_2_direction', 'uav_2_speed', 
            'uav_3_direction', 'uav_3_speed',
            'uav_4_direction', 'uav_4_speed',
            'uav_5_direction', 'uav_5_speed',
            # 每架无人机3枚烟幕弹的参数
            'uav_1_smoke_1_deploy_time', 'uav_1_smoke_1_explode_delay',
            'uav_1_smoke_2_deploy_time', 'uav_1_smoke_2_explode_delay',
            'uav_1_smoke_3_deploy_time', 'uav_1_smoke_3_explode_delay',
            'uav_2_smoke_1_deploy_time', 'uav_2_smoke_1_explode_delay',
            'uav_2_smoke_2_deploy_time', 'uav_2_smoke_2_explode_delay',
            'uav_2_smoke_3_deploy_time', 'uav_2_smoke_3_explode_delay',
            'uav_3_smoke_1_deploy_time', 'uav_3_smoke_1_explode_delay',
            'uav_3_smoke_2_deploy_time', 'uav_3_smoke_2_explode_delay',
            'uav_3_smoke_3_deploy_time', 'uav_3_smoke_3_explode_delay',
            'uav_4_smoke_1_deploy_time', 'uav_4_smoke_1_explode_delay',
            'uav_4_smoke_2_deploy_time', 'uav_4_smoke_2_explode_delay',
            'uav_4_smoke_3_deploy_time', 'uav_4_smoke_3_explode_delay',
            'uav_5_smoke_1_deploy_time', 'uav_5_smoke_1_explode_delay',
            'uav_5_smoke_2_deploy_time', 'uav_5_smoke_2_explode_delay',
            'uav_5_smoke_3_deploy_time', 'uav_5_smoke_3_explode_delay'
        ]
        
        # 创建参数字典
        params = {name: individual[i] for i, name in enumerate(param_names)}
        
        # 计算综合遮蔽效果
        total_masking_score = calculate_problem5_multi_uav_multi_missile_masking(**params)
        
        return total_masking_score
        
    except Exception as e:
        logger.error(f"个体评估失败: {e}")
        return -1000.0


class Problem5DifferentialEvolution:
    """第五问差分进化优化器"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.bounds = calculate_problem5_bounds()
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)  # 应该是40维（5*2 + 5*3*2）
        
        logger.info(f"第五问优化器初始化:")
        logger.info(f"  决策变量维度: {self.n_dims}")
        logger.info(f"  种群规模: {self.config.population_size}")
        logger.info(f"  最大代数: {self.config.max_generations}")
        
        # 性能统计
        self.performance_stats = {
            'generation_times': [],
            'best_fitness_history': [],
            'avg_fitness_history': []
        }
        
        # 设置并行
        if self.config.use_parallel:
            self.n_processes = min(mp.cpu_count(), self.config.population_size)
            logger.info(f"  并行进程数: {self.n_processes}")
    
    def _initialize_population(self) -> np.ndarray:
        """初始化种群"""
        population = np.random.rand(self.config.population_size, self.n_dims)
        for i, (lower, upper) in enumerate(self.bounds_list):
            population[:, i] = population[:, i] * (upper - lower) + lower
        return population
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.zeros(self.config.population_size)
        
        if self.config.use_parallel:
            # 并行评估
            individual_data = [(population[i], self.bounds_list) for i in range(self.config.population_size)]
            
            with ProcessPoolExecutor(max_workers=self.n_processes) as executor:
                future_to_idx = {
                    executor.submit(evaluate_individual_problem5, data): i 
                    for i, data in enumerate(individual_data)
                }
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        fitness[idx] = future.result()
                    except Exception as e:
                        logger.error(f"个体{idx}评估失败: {e}")
                        fitness[idx] = -1000.0
        else:
            # 串行评估
            for i in range(self.config.population_size):
                fitness[i] = evaluate_individual_problem5((population[i], self.bounds_list))
        
        return fitness
    
    def _generate_trials(self, population: np.ndarray) -> np.ndarray:
        """生成试验向量"""
        trials = np.zeros_like(population)
        
        for i in range(self.config.population_size):
            # 选择三个不同的个体
            candidates = list(range(self.config.population_size))
            candidates.remove(i)
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            
            # DE/rand/1 变异
            mutant = population[r1] + self.config.F * (population[r2] - population[r3])
            
            # 边界处理
            for j, (lower, upper) in enumerate(self.bounds_list):
                if mutant[j] < lower:
                    mutant[j] = lower + np.random.rand() * (upper - lower) * 0.1
                elif mutant[j] > upper:
                    mutant[j] = upper - np.random.rand() * (upper - lower) * 0.1
            
            # 交叉
            trial = population[i].copy()
            crossover_mask = np.random.rand(self.n_dims) < self.config.CR
            crossover_mask[np.random.randint(self.n_dims)] = True  # 确保至少一个基因交叉
            trial[crossover_mask] = mutant[crossover_mask]
            
            trials[i] = trial
        
        return trials
    
    def optimize(self) -> Dict[str, Any]:
        """执行优化"""
        logger.info("="*80)
        logger.info("🚀 启动第五问多导弹协同遮蔽优化")
        logger.info("="*80)
        
        # 初始化种群
        population = self._initialize_population()
        fitness = self._evaluate_population(population)
        
        # 记录最优个体
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_individual = population[best_idx].copy()
        
        logger.info(f"初始种群最优适应度: {best_fitness:.6f}")
        
        # 主优化循环
        for generation in range(self.config.max_generations):
            generation_start = time.time()
            
            # 生成试验向量
            trials = self._generate_trials(population)
            trial_fitness = self._evaluate_population(trials)
            
            # 选择操作
            for i in range(self.config.population_size):
                if trial_fitness[i] > fitness[i]:
                    population[i] = trials[i]
                    fitness[i] = trial_fitness[i]
                    
                    if trial_fitness[i] > best_fitness:
                        best_fitness = trial_fitness[i]
                        best_individual = trials[i].copy()
            
            generation_time = time.time() - generation_start
            
            # 记录统计信息
            self.performance_stats['generation_times'].append(generation_time)
            self.performance_stats['best_fitness_history'].append(best_fitness)
            self.performance_stats['avg_fitness_history'].append(np.mean(fitness))
            
            # 输出进度
            if generation % 10 == 0 or generation < 10:
                avg_fitness = np.mean(fitness)
                print(f"第{generation:4d}代 | 最优: {best_fitness:8.4f} | "
                      f"平均: {avg_fitness:8.4f} | 时间: {generation_time:6.2f}s")
        
        logger.info("="*80)
        logger.info("🎯 优化完成!")
        logger.info(f"最优综合遮蔽时长: {best_fitness:.6f}秒")
        
        return {
            'best_fitness': best_fitness,
            'best_individual': best_individual,
            'performance_stats': self.performance_stats
        }


def decode_solution(best_individual: np.ndarray, bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """解码最优解"""
    param_names = list(bounds.keys())
    solution = {name: best_individual[i] for i, name in enumerate(param_names)}
    
    # 按无人机组织结果
    result = {
        'uavs': {},
        'summary': {}
    }
    
    # 解析每架无人机的参数
    for uav_id in range(1, 6):  # UAV 1-5
        uav_key = f'UAV{uav_id}'
        result['uavs'][uav_key] = {
            'direction': solution[f'uav_{uav_id}_direction'],
            'speed': solution[f'uav_{uav_id}_speed'],
            'smokes': []
        }
        
        # 每架无人机的3枚烟幕弹
        for smoke_id in range(1, 4):  # Smoke 1-3
            smoke_info = {
                'deploy_time': solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time'],
                'explode_delay': solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay'],
                'explode_time': solution[f'uav_{uav_id}_smoke_{smoke_id}_deploy_time'] + 
                               solution[f'uav_{uav_id}_smoke_{smoke_id}_explode_delay']
            }
            result['uavs'][uav_key]['smokes'].append(smoke_info)
    
    return result


def save_results_to_excel(result: Dict[str, Any], decoded_solution: Dict[str, Any], 
                         filename: str = "附件/result3.xlsx"):
    """保存结果到Excel文件"""
    
    # 创建多个工作表
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        
        # 工作表1：优化结果摘要
        summary_data = {
            '指标': ['最优综合遮蔽时长(秒)', '使用无人机数量', '使用烟幕弹总数', '优化代数'],
            '数值': [
                result['best_fitness'],
                5,  # 5架无人机
                15, # 每架3枚，共15枚
                len(result['performance_stats']['best_fitness_history'])
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='优化结果摘要', index=False)
        
        # 工作表2：无人机投放策略详情
        strategy_rows = []
        for uav_name, uav_data in decoded_solution['uavs'].items():
            base_row = {
                '无人机': uav_name,
                '飞行方向(度)': f"{uav_data['direction']:.2f}",
                '飞行速度(m/s)': f"{uav_data['speed']:.2f}"
            }
            
            for i, smoke in enumerate(uav_data['smokes'], 1):
                row = base_row.copy()
                row.update({
                    f'烟幕弹{i}_投放时间(s)': f"{smoke['deploy_time']:.3f}",
                    f'烟幕弹{i}_起爆延时(s)': f"{smoke['explode_delay']:.3f}",
                    f'烟幕弹{i}_起爆时间(s)': f"{smoke['explode_time']:.3f}"
                })
                if i == 1:  # 只在第一行显示无人机信息
                    strategy_rows.append(row)
                else:
                    # 其他行只显示烟幕弹信息
                    smoke_only_row = {k: v if '烟幕弹' in k else '' for k, v in row.items()}
                    smoke_only_row['无人机'] = ''
                    strategy_rows.append(smoke_only_row)
        
        pd.DataFrame(strategy_rows).to_excel(writer, sheet_name='投放策略详情', index=False)
        
        # 工作表3：时间序列分析
        time_analysis = []
        all_events = []
        
        for uav_name, uav_data in decoded_solution['uavs'].items():
            for i, smoke in enumerate(uav_data['smokes'], 1):
                all_events.append({
                    '时间(s)': smoke['deploy_time'],
                    '事件': f"{uav_name}_烟幕弹{i}_投放",
                    '无人机': uav_name
                })
                all_events.append({
                    '时间(s)': smoke['explode_time'],
                    '事件': f"{uav_name}_烟幕弹{i}_起爆",
                    '无人机': uav_name
                })
        
        # 按时间排序
        all_events.sort(key=lambda x: x['时间(s)'])
        pd.DataFrame(all_events).to_excel(writer, sheet_name='时间序列分析', index=False)
        
        # 工作表4：收敛历史
        convergence_data = {
            '代数': list(range(len(result['performance_stats']['best_fitness_history']))),
            '最优适应度': result['performance_stats']['best_fitness_history'],
            '平均适应度': result['performance_stats']['avg_fitness_history']
        }
        pd.DataFrame(convergence_data).to_excel(writer, sheet_name='收敛历史', index=False)
    
    logger.info(f"结果已保存到: {filename}")


def main():
    """主函数"""
    print("🎯 CUMCM2025 Problem 5 - 多无人机多导弹协同烟幕遮蔽优化")
    print("="*80)
    print("任务：5架无人机，每架最多3枚烟幕弹，对抗3枚来袭导弹")
    print("目标：最大化综合遮蔽时长")
    print("="*80)
    
    # 创建优化配置
    config = OptimizationConfig(
        population_size=120,      # 大种群以应对高维问题
        max_generations=800,      # 充分迭代
        F=0.5,
        CR=0.9,
        use_parallel=True
    )
    
    # 创建优化器
    optimizer = Problem5DifferentialEvolution(config)
    
    # 执行优化
    start_time = time.time()
    result = optimizer.optimize()
    end_time = time.time()
    
    print(f"\n优化完成，总用时: {end_time - start_time:.2f} 秒")
    print(f"最优综合遮蔽时长: {result['best_fitness']:.6f} 秒")
    
    # 解码解决方案
    decoded_solution = decode_solution(result['best_individual'], optimizer.bounds)
    
    # 打印关键结果
    print("\n🎯 最优投放策略:")
    for uav_name, uav_data in decoded_solution['uavs'].items():
        print(f"\n{uav_name}:")
        print(f"  飞行方向: {uav_data['direction']:.2f}°")
        print(f"  飞行速度: {uav_data['speed']:.2f} m/s")
        print("  烟幕弹投放:")
        for i, smoke in enumerate(uav_data['smokes'], 1):
            print(f"    烟幕弹{i}: {smoke['deploy_time']:.3f}s投放, "
                  f"{smoke['explode_delay']:.3f}s延时, {smoke['explode_time']:.3f}s起爆")
    
    # 保存结果
    save_results_to_excel(result, decoded_solution)
    
    print(f"\n✅ 结果已保存到 附件/result3.xlsx")
    
    return result


if __name__ == "__main__":
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)
    main() 