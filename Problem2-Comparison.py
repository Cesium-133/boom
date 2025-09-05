"""
问题2综合比较：PSO vs GA算法性能对比

同时运行粒子群优化(PSO)和遗传算法(GA)，
自动进行结果对比和交叉验证。
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
from typing import Dict, Any

# 导入两种优化器
import sys
import importlib.util

# 动态导入PSO和GA模块
def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# 导入优化器
pso_module = import_module_from_path("pso_optimizer", "Problem2-PSO.py")
ga_module = import_module_from_path("ga_optimizer", "Problem2-GA.py")

# 配置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run_pso_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """运行PSO优化"""
    print("="*60)
    print("运行粒子群优化算法 (PSO)")
    print("="*60)
    
    optimizer = pso_module.PSO_Optimizer(**params)
    
    start_time = time.time()
    best_position, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    # 解码最优解
    keys = list(optimizer.bounds.keys())
    best_params = {keys[i]: best_position[i] for i in range(len(keys))}
    
    results = {
        'algorithm': 'PSO',
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'algorithm_params': params,
        'bounds': optimizer.bounds,
        'fitness_history': optimizer.fitness_history,
        'optimizer': optimizer
    }
    
    return results


def run_ga_optimization(params: Dict[str, Any]) -> Dict[str, Any]:
    """运行GA优化"""
    print("="*60)
    print("运行遗传算法 (GA)")
    print("="*60)
    
    optimizer = ga_module.GA_Optimizer(**params)
    
    start_time = time.time()
    best_individual, best_fitness = optimizer.optimize()
    end_time = time.time()
    
    # 解码最优解
    keys = list(optimizer.bounds.keys())
    best_params = {keys[i]: best_individual[i] for i in range(len(keys))}
    
    results = {
        'algorithm': 'GA',
        'best_params': best_params,
        'best_fitness': best_fitness,
        'optimization_time': end_time - start_time,
        'algorithm_params': params,
        'bounds': optimizer.bounds,
        'best_fitness_history': optimizer.best_fitness_history,
        'avg_fitness_history': optimizer.avg_fitness_history,
        'optimizer': optimizer
    }
    
    return results


def compare_results(pso_results: Dict[str, Any], ga_results: Dict[str, Any]):
    """比较两种算法的结果"""
    print("="*80)
    print("算法结果对比分析")
    print("="*80)
    
    pso_params = pso_results['best_params']
    ga_params = ga_results['best_params']
    pso_fitness = pso_results['best_fitness']
    ga_fitness = ga_results['best_fitness']
    
    print(f"\n{'参数':<15} {'PSO结果':<12} {'GA结果':<12} {'绝对差异':<12} {'相对差异(%)':<15}")
    print("-" * 75)
    
    # 比较各个参数
    param_names = {
        'v_FY1': '速度 (m/s)',
        'theta_FY1': '方向 (度)',
        't_deploy': '投放时间 (s)',
        't_fuse': '引信延时 (s)'
    }
    
    for key, name in param_names.items():
        pso_val = pso_params[key]
        ga_val = ga_params[key]
        abs_diff = abs(pso_val - ga_val)
        rel_diff = (abs_diff / max(abs(pso_val), abs(ga_val))) * 100 if max(abs(pso_val), abs(ga_val)) > 0 else 0
        
        print(f"{name:<15} {pso_val:<12.3f} {ga_val:<12.3f} {abs_diff:<12.3f} {rel_diff:<15.2f}")
    
    # 比较适应度
    fitness_diff = abs(pso_fitness - ga_fitness)
    fitness_rel_diff = (fitness_diff / max(pso_fitness, ga_fitness)) * 100
    
    print(f"{'遮蔽时长 (s)':<15} {pso_fitness:<12.6f} {ga_fitness:<12.6f} {fitness_diff:<12.6f} {fitness_rel_diff:<15.2f}")
    
    # 比较计算时间
    pso_time = pso_results['optimization_time']
    ga_time = ga_results['optimization_time']
    time_diff = abs(pso_time - ga_time)
    
    print(f"{'计算时间 (s)':<15} {pso_time:<12.2f} {ga_time:<12.2f} {time_diff:<12.2f} {'':<15}")
    
    # 一致性分析
    print(f"\n{'='*60}")
    print("一致性分析:")
    
    if fitness_diff < 0.001:
        consistency = "✅ 高度一致"
        print(f"  {consistency} - 两种算法找到了几乎相同的最优解")
    elif fitness_diff < 0.01:
        consistency = "⚠️  基本一致"
        print(f"  {consistency} - 两种算法结果相近，存在小幅差异")
    elif fitness_diff < 0.1:
        consistency = "⚠️  存在差异"
        print(f"  {consistency} - 两种算法结果有一定差异，建议进一步验证")
    else:
        consistency = "❌ 差异较大"
        print(f"  {consistency} - 两种算法结果差异较大，需要检查算法参数")
    
    # 性能分析
    print(f"\n性能分析:")
    if pso_fitness > ga_fitness:
        print(f"  PSO找到了更优的解，适应度高出 {fitness_diff:.6f}")
    elif ga_fitness > pso_fitness:
        print(f"  GA找到了更优的解，适应度高出 {fitness_diff:.6f}")
    else:
        print(f"  两种算法找到了相同的最优解")
    
    if pso_time < ga_time:
        print(f"  PSO运行更快，节省时间 {time_diff:.2f}秒")
    elif ga_time < pso_time:
        print(f"  GA运行更快，节省时间 {time_diff:.2f}秒")
    else:
        print(f"  两种算法运行时间相近")
    
    return {
        'fitness_difference': fitness_diff,
        'fitness_relative_difference': fitness_rel_diff,
        'time_difference': time_diff,
        'consistency': consistency,
        'better_algorithm': 'PSO' if pso_fitness > ga_fitness else 'GA' if ga_fitness > pso_fitness else 'Equal',
        'faster_algorithm': 'PSO' if pso_time < ga_time else 'GA' if ga_time < pso_time else 'Equal'
    }


def plot_comparison(pso_results: Dict[str, Any], ga_results: Dict[str, Any]):
    """绘制对比图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 收敛曲线对比
    ax1 = axes[0, 0]
    ax1.plot(pso_results['fitness_history'], 'b-', linewidth=2, label='PSO')
    ax1.plot(ga_results['best_fitness_history'], 'r-', linewidth=2, label='GA最佳')
    ax1.plot(ga_results['avg_fitness_history'], 'r--', linewidth=1, label='GA平均')
    ax1.set_title('收敛曲线对比')
    ax1.set_xlabel('迭代次数/代数')
    ax1.set_ylabel('适应度')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 参数对比雷达图
    ax2 = axes[0, 1]
    categories = ['速度', '方向', '投放时间', '引信延时']
    
    # 标准化参数值到0-1范围
    pso_params = pso_results['best_params']
    ga_params = ga_results['best_params']
    bounds = pso_results['bounds']
    
    pso_normalized = []
    ga_normalized = []
    
    for key in ['v_FY1', 'theta_FY1', 't_deploy', 't_fuse']:
        min_val, max_val = bounds[key]
        pso_norm = (pso_params[key] - min_val) / (max_val - min_val)
        ga_norm = (ga_params[key] - min_val) / (max_val - min_val)
        pso_normalized.append(pso_norm)
        ga_normalized.append(ga_norm)
    
    # 绘制雷达图
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    pso_normalized.append(pso_normalized[0])
    ga_normalized.append(ga_normalized[0])
    
    ax2 = plt.subplot(2, 3, 2, projection='polar')
    ax2.plot(angles, pso_normalized, 'b-', linewidth=2, label='PSO')
    ax2.fill(angles, pso_normalized, 'blue', alpha=0.25)
    ax2.plot(angles, ga_normalized, 'r-', linewidth=2, label='GA')
    ax2.fill(angles, ga_normalized, 'red', alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_title('参数对比雷达图')
    ax2.legend()
    
    # 3. 性能指标对比
    ax3 = axes[0, 2]
    metrics = ['适应度', '计算时间(s)']
    pso_values = [pso_results['best_fitness'], pso_results['optimization_time']]
    ga_values = [ga_results['best_fitness'], ga_results['optimization_time']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # 标准化显示
    pso_display = [pso_values[0], pso_values[1]/10]  # 时间除以10便于显示
    ga_display = [ga_values[0], ga_values[1]/10]
    
    ax3.bar(x - width/2, pso_display, width, label='PSO', color='blue', alpha=0.7)
    ax3.bar(x + width/2, ga_display, width, label='GA', color='red', alpha=0.7)
    ax3.set_title('性能指标对比')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['适应度', '计算时间(×10)'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数分布对比
    ax4 = axes[1, 0]
    param_keys = list(pso_params.keys())
    pso_vals = [pso_params[key] for key in param_keys]
    ga_vals = [ga_params[key] for key in param_keys]
    
    x = np.arange(len(param_keys))
    ax4.scatter(x, pso_vals, color='blue', s=100, label='PSO', marker='o')
    ax4.scatter(x, ga_vals, color='red', s=100, label='GA', marker='s')
    
    for i in range(len(param_keys)):
        ax4.plot([i, i], [pso_vals[i], ga_vals[i]], 'k--', alpha=0.5)
    
    ax4.set_title('参数值对比')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['速度', '方向', '投放时间', '引信延时'], rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 算法收敛速度对比
    ax5 = axes[1, 1]
    if len(pso_results['fitness_history']) > 1:
        pso_improvements = np.diff(pso_results['fitness_history'])
        ax5.plot(pso_improvements, 'b-', linewidth=2, label='PSO改进速度')
    
    if len(ga_results['best_fitness_history']) > 1:
        ga_improvements = np.diff(ga_results['best_fitness_history'])
        ax5.plot(ga_improvements, 'r-', linewidth=2, label='GA改进速度')
    
    ax5.set_title('收敛速度对比')
    ax5.set_xlabel('迭代次数')
    ax5.set_ylabel('适应度改进量')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 最终结果对比
    ax6 = axes[1, 2]
    algorithms = ['PSO', 'GA']
    fitness_values = [pso_results['best_fitness'], ga_results['best_fitness']]
    colors = ['blue', 'red']
    
    bars = ax6.bar(algorithms, fitness_values, color=colors, alpha=0.7)
    ax6.set_title('最终适应度对比')
    ax6.set_ylabel('有效遮蔽时长 (s)')
    
    # 在柱状图上添加数值标签
    for bar, value in zip(bars, fitness_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.6f}', ha='center', va='bottom')
    
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def save_results(pso_results: Dict[str, Any], ga_results: Dict[str, Any], comparison: Dict[str, Any]):
    """保存结果到文件"""
    # 准备保存的数据（移除不能序列化的对象）
    save_data = {
        'pso': {
            'algorithm': pso_results['algorithm'],
            'best_params': pso_results['best_params'],
            'best_fitness': pso_results['best_fitness'],
            'optimization_time': pso_results['optimization_time'],
            'algorithm_params': pso_results['algorithm_params'],
            'bounds': pso_results['bounds'],
            'fitness_history': pso_results['fitness_history']
        },
        'ga': {
            'algorithm': ga_results['algorithm'],
            'best_params': ga_results['best_params'],
            'best_fitness': ga_results['best_fitness'],
            'optimization_time': ga_results['optimization_time'],
            'algorithm_params': ga_results['algorithm_params'],
            'bounds': ga_results['bounds'],
            'best_fitness_history': ga_results['best_fitness_history'],
            'avg_fitness_history': ga_results['avg_fitness_history']
        },
        'comparison': comparison,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # 保存为JSON文件
    filename = f"optimization_comparison_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到文件: {filename}")


def main():
    """主函数"""
    print("问题2：PSO vs GA算法性能对比")
    print("="*60)
    
    # 设置算法参数
    pso_params = {
        'n_particles': 30,
        'max_iterations': 50,  # 减少迭代次数以便快速对比
        'w_start': 0.9,
        'w_end': 0.4,
        'c1': 2.0,
        'c2': 2.0
    }
    
    ga_params = {
        'population_size': 30,
        'max_generations': 50,  # 减少代数以便快速对比
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'elite_size': 2,
        'tournament_size': 3,
        'use_parallel': True
    }
    
    print(f"PSO参数: {pso_params}")
    print(f"GA参数: {ga_params}")
    
    # 运行两种算法
    try:
        pso_results = run_pso_optimization(pso_params)
        ga_results = run_ga_optimization(ga_params)
        
        # 比较结果
        comparison = compare_results(pso_results, ga_results)
        
        # 绘制对比图表
        plot_comparison(pso_results, ga_results)
        
        # 保存结果
        save_results(pso_results, ga_results, comparison)
        
        # 输出最终建议
        print(f"\n{'='*60}")
        print("最终建议:")
        
        if comparison['better_algorithm'] == 'Equal':
            print("✅ 两种算法表现相当，结果可信度高")
        else:
            better_alg = comparison['better_algorithm']
            print(f"🏆 {better_alg} 算法找到了更优的解")
        
        if comparison['consistency'] == "✅ 高度一致":
            print("✅ 算法结果高度一致，建议采用找到的最优解")
        else:
            print("⚠️  建议增加迭代次数或调整算法参数以获得更稳定的结果")
        
        return pso_results, ga_results, comparison
        
    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


if __name__ == "__main__":
    pso_results, ga_results, comparison = main() 