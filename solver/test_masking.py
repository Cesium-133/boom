"""
遮蔽计算模块简单测试

验证遮蔽计算模块的基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import (
    calculate_single_uav_single_smoke_masking,
    calculate_single_uav_triple_smoke_masking,
    calculate_multi_uav_single_smoke_masking
)


def test_single_uav_single_smoke():
    """测试单无人机单烟幕弹场景"""
    print("=== 单无人机单烟幕弹测试 ===")
    
    # 测试参数
    uav_direction = 90      # 向北飞行
    uav_speed = 120         # 120 m/s
    smoke_deploy_time = 5.0 # 5秒时投放
    smoke_explode_delay = 1.5 # 1.5秒后起爆
    
    try:
        duration = calculate_single_uav_single_smoke_masking(
            uav_direction, uav_speed, smoke_deploy_time, smoke_explode_delay
        )
        print(f"输入参数:")
        print(f"  无人机方向: {uav_direction}°")
        print(f"  无人机速度: {uav_speed} m/s")
        print(f"  烟幕弹投放时间: {smoke_deploy_time} s")
        print(f"  起爆延时: {smoke_explode_delay} s")
        print(f"结果: 有效遮蔽时长 = {duration:.3f} 秒")
        return True
    except Exception as e:
        print(f"计算出错: {e}")
        return False


def test_basic_import():
    """测试基本导入功能"""
    print("=== 基本导入测试 ===")
    try:
        from config import CONSTANTS, TARGETS
        from geometry import distance_between, Vector3
        from trajectory import TrajectoryCalculator
        
        from config import SMOKE_PARAMS, MISSILES
        
        print("配置常量:")
        print(f"  重力加速度: {CONSTANTS['g']} m/s²")
        print(f"  烟幕云半径: {SMOKE_PARAMS['cloud_radius']} m")
        print(f"  M1导弹速度: {MISSILES['M1']['speed']} m/s")
        
        print("目标配置:")
        print(f"  虚假目标: {TARGETS['fake_target']}")
        print(f"  真实目标中心: {TARGETS['true_target']['base_center']}")
        
        # 测试几何计算
        p1: Vector3 = (0, 0, 0)
        p2: Vector3 = (3, 4, 0)
        dist = distance_between(p1, p2)
        print(f"几何计算测试: 点(0,0,0)到点(3,4,0)的距离 = {dist}")
        
        # 测试轨迹计算器
        traj_calc = TrajectoryCalculator()
        missile_traj = traj_calc.create_missile_trajectory("M1")
        pos_at_0 = missile_traj(0)
        print(f"导弹初始位置: {pos_at_0}")
        
        print("基本导入测试通过！")
        return True
    except Exception as e:
        print(f"导入测试失败: {e}")
        return False


if __name__ == "__main__":
    print("导弹拦截烟幕遮蔽分析模块测试")
    print("=" * 50)
    
    # 运行测试
    test_results = []
    test_results.append(test_basic_import())
    print()
    test_results.append(test_single_uav_single_smoke())
    
    # 总结
    passed = sum(test_results)
    total = len(test_results)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✅ 所有测试通过！模块可以正常使用。")
    else:
        print("❌ 部分测试失败，请检查模块实现。")
