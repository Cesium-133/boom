"""
性能对比测试脚本
"""

import time
import numpy as np
from solver import calculate_single_uav_single_smoke_masking

try:
    from solver.core_optimized import calculate_single_uav_single_smoke_masking_optimized
    HAS_OPTIMIZED = True
    print("优化版本可用")
except ImportError:
    HAS_OPTIMIZED = False

def benchmark_performance():
    """性能对比测试"""
    # 测试参数
    test_params = [
        (120.0, 90.0, 1.5, 3.6),
        (140.0, 45.0, 2.0, 4.0),
        (100.0, 180.0, 1.0, 2.5),
        (130.0, 270.0, 1.8, 3.2),
        (110.0, 315.0, 2.2, 3.8),
    ]
    
    n_runs = 10  # 每个参数组合运行的次数
    
    print("="*60)
    print("性能对比测试")
    print("="*60)
    
    # 测试标准版本
    print("测试标准版本...")
    start_time = time.time()
    standard_results = []
    
    for params in test_params:
        for _ in range(n_runs):
            result = calculate_single_uav_single_smoke_masking(*params)
            standard_results.append(result)
    
    standard_time = time.time() - start_time
    print(f"标准版本总耗时: {standard_time:.4f} 秒")
    
    if HAS_OPTIMIZED:
        # 测试优化版本
        print("测试优化版本...")
        start_time = time.time()
        optimized_results = []
        
        for params in test_params:
            for _ in range(n_runs):
                result = calculate_single_uav_single_smoke_masking_optimized(*params)
                optimized_results.append(result)
        
        optimized_time = time.time() - start_time
        print(f"优化版本总耗时: {optimized_time:.4f} 秒")
        
        # 计算加速比
        speedup = standard_time / optimized_time
        print(f"加速比: {speedup:.2f}x")
        
        # 验证结果一致性
        max_diff = max(abs(s - o) for s, o in zip(standard_results, optimized_results))
        print(f"最大结果差异: {max_diff:.6f}")
        
        if max_diff < 1e-6:
            print("✅ 结果验证通过")
        else:
            print("❌ 结果验证失败")
    else:
        print("❌ 优化版本不可用，请安装 numba")

if __name__ == "__main__":
    benchmark_performance()