"""
导弹拦截烟幕遮蔽分析求解器模块

提供导弹拦截过程中烟幕弹有效遮蔽时长的计算功能。
"""

from .core import (
    calculate_single_uav_single_smoke_masking,
    calculate_single_uav_triple_smoke_masking,
    calculate_multi_uav_single_smoke_masking,
    calculate_single_uav_triple_smoke_masking_multiple,
    calculate_multi_uav_single_smoke_masking_multiple
)

from .config import CONSTANTS, TARGETS, MISSILES, UAVS, SMOKE_PARAMS

__all__ = [
    'calculate_single_uav_single_smoke_masking',
    'calculate_single_uav_triple_smoke_masking', 
    'calculate_multi_uav_single_smoke_masking',
    'calculate_single_uav_triple_smoke_masking_multiple',
    'calculate_multi_uav_single_smoke_masking_multiple',
    'CONSTANTS',
    'TARGETS',
    'MISSILES',
    'UAVS',
    'SMOKE_PARAMS'
]