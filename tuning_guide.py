"""
差分进化算法渐进式调参指南

从4.588逼近4.8的系统化调参策略
"""

def get_tuning_stages():
    """渐进式调参阶段"""
    
    stages = {
        "阶段1 - 基础增强": {
            "目标": "4.588 → 4.65",
            "策略": "增大种群和代数",
            "参数": {
                'population_size': 80,
                'max_generations': 800,
                'F_min': 0.3, 'F_max': 1.0,
                'CR_min': 0.1, 'CR_max': 0.9,
            },
            "预期提升": "0.06"
        },
        
        "阶段2 - 参数优化": {
            "目标": "4.65 → 4.72", 
            "策略": "优化F和CR范围",
            "参数": {
                'population_size': 100,
                'max_generations': 1000,
                'F_min': 0.2, 'F_max': 1.3,
                'CR_min': 0.05, 'CR_max': 0.95,
                'restart_threshold': 40,
            },
            "预期提升": "0.07"
        },
        
        "阶段3 - 机制增强": {
            "目标": "4.72 → 4.78",
            "策略": "添加局部搜索和重启",
            "参数": {
                'population_size': 120,
                'max_generations': 1200,
                'F_min': 0.15, 'F_max': 1.5,
                'CR_min': 0.02, 'CR_max': 0.98,
                'restart_threshold': 30,
                'local_search_prob': 0.12,
                'multi_population': True,
            },
            "预期提升": "0.06"
        },
        
        "阶段4 - 精细调优": {
            "目标": "4.78 → 4.8+",
            "策略": "极限参数+多重机制",
            "参数": {
                'population_size': 150,
                'max_generations': 1500,
                'F_min': 0.1, 'F_max': 2.0,
                'CR_min': 0.01, 'CR_max': 0.99,
                'restart_threshold': 25,
                'local_search_prob': 0.2,
                'multi_population': True,
                'adaptive_parameters': True,
            },
            "预期提升": "0.02+"
        }
    }
    
    return stages

def print_tuning_recommendations():
    """打印调参建议"""
    print("🎯 从4.588到4.8的调参路线图")
    print("="*50)
    
    stages = get_tuning_stages()
    
    for stage_name, stage_info in stages.items():
        print(f"\n📍 {stage_name}")
        print(f"   目标: {stage_info['目标']}")
        print(f"   策略: {stage_info['策略']}")
        print(f"   预期提升: {stage_info['预期提升']}")
        print(f"   关键参数:")
        for param, value in stage_info['参数'].items():
            print(f"     {param}: {value}")

def get_emergency_params():
    """应急调参方案 - 如果常规方法不行"""
    return {
        "超大种群方案": {
            'population_size': 200,
            'max_generations': 2000,
            'F_min': 0.05, 'F_max': 2.5,
            'CR_min': 0.01, 'CR_max': 0.99,
            'restart_threshold': 20,
            'local_search_prob': 0.25,
            'multi_population': True,
            'n_subpopulations': 8,
        },
        
        "多次重启方案": {
            'population_size': 100,
            'max_generations': 500,  # 短代数
            'restart_threshold': 15,  # 频繁重启
            'n_runs': 10,  # 运行10次取最好
        },
        
        "混合策略方案": {
            # 前期大步长全局搜索
            'phase1': {'F_min': 0.8, 'F_max': 2.0, 'generations': 300},
            # 后期小步长精细搜索  
            'phase2': {'F_min': 0.1, 'F_max': 0.5, 'generations': 700},
        }
    }

# 实时监控指标
def monitoring_metrics():
    """需要监控的关键指标"""
    return {
        "收敛速度": "fitness_history的斜率",
        "种群多样性": "diversity_history",
        "停滞检测": "连续多少代无改进",
        "参数效果": "F和CR的历史变化",
        "重启效果": "重启前后的适应度跳跃",
    }

if __name__ == "__main__":
    print_tuning_recommendations()
    
    print("\n" + "="*50)
    print("🚨 应急方案（如果上述方案效果不佳）")
    emergency = get_emergency_params()
    for name, params in emergency.items():
        print(f"\n{name}:")
        for key, value in params.items():
            print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("📊 关键监控指标")
    metrics = monitoring_metrics()
    for metric, desc in metrics.items():
        print(f"  {metric}: {desc}") 