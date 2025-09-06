import os
import sys
import time
import math
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import importlib.util

# -----------------------------------------------------------------------------
# 动态加载 Problem2-DE-Tuned.py（文件名包含连字符，不能直接 import）
# -----------------------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_PATH = os.path.join(ROOT, 'Problem2-DE-Tuned.py')
MODULE_NAME = 'problem2_de_tuned_dynamic'

spec = importlib.util.spec_from_file_location(MODULE_NAME, FILE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError('无法加载 Problem2-DE-Tuned.py')
mod = importlib.util.module_from_spec(spec)
sys.modules[MODULE_NAME] = mod
spec.loader.exec_module(mod)

TunedDifferentialEvolution = mod.TunedDifferentialEvolution
calculate_bounds = mod.calculate_bounds
evaluate_individual_fitness = mod.evaluate_individual_fitness

# -----------------------------------------------------------------------------
# Streamlit 页面配置
# -----------------------------------------------------------------------------
st.set_page_config(page_title='DE 调参与可视化', layout='wide')
st.title('问题2 - 差分进化调参与可视化')

# -----------------------------------------------------------------------------
# 侧边栏参数
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header('算法参数')
    population_size = st.slider('种群规模', 20, 200, 80, step=10)
    max_generations = st.slider('最大代数', 50, 2000, 500, step=50)
    F_min = st.slider('F 最小值', 0.1, 1.0, 0.3, step=0.05)
    F_max = st.slider('F 最大值', 0.4, 2.0, 1.2, step=0.05)
    CR_min = st.slider('CR 最小值', 0.01, 0.5, 0.05, step=0.01)
    CR_max = st.slider('CR 最大值', 0.5, 1.0, 0.95, step=0.01)
    restart_threshold = st.slider('重启阈值(停滞代)', 10, 200, 50, step=5)
    local_search_prob = st.slider('局部搜索概率', 0.0, 0.5, 0.1, step=0.01)
    multi_population = st.checkbox('启用多子种群', True)
    # Windows 下默认关闭并行，避免多进程在动态模块上的导入问题
    use_parallel = st.checkbox('并行计算(Windows建议关闭)', False)

    st.header('搜索空间边界')
    try:
        bounds: Dict[str, Tuple[float, float]] = calculate_bounds()
        st.json(bounds)
    except Exception as e:
        st.error(f'计算边界失败: {e}')
        st.stop()

    start_btn = st.button('开始优化', type='primary')
    stop_btn = st.button('停止')

# -----------------------------------------------------------------------------
# 运行状态
# -----------------------------------------------------------------------------
if 'stop_flag' not in st.session_state:
    st.session_state.stop_flag = False
if stop_btn:
    st.session_state.stop_flag = True

# -----------------------------------------------------------------------------
# 布局与占位
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    fitness_chart = st.line_chart(pd.DataFrame({'best_fitness': []}))
with col2:
    diversity_chart = st.line_chart(pd.DataFrame({'diversity': []}))

meta_area = st.container()
scatter_area = st.container()
local_area = st.expander('局部最优性分析', expanded=True)

# -----------------------------------------------------------------------------
# 局部最优性热力图函数
# -----------------------------------------------------------------------------
PARAM_NAMES: List[str] = ['v_FY1', 'theta_FY1', 't_deploy', 't_fuse']

@st.cache_data(show_spinner=False)
def compute_local_heatmap(best_position: Tuple[float, float, float, float],
                          bounds_list: List[Tuple[float, float]],
                          x_idx: int,
                          y_idx: int,
                          radius_ratio: float,
                          grid_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    best = np.array(best_position, dtype=float)
    x_min, x_max = bounds_list[x_idx]
    y_min, y_max = bounds_list[y_idx]
    # 半径按边界范围的比例定义
    x_rad = (x_max - x_min) * radius_ratio
    y_rad = (y_max - y_min) * radius_ratio

    xs = np.linspace(max(x_min, best[x_idx] - x_rad), min(x_max, best[x_idx] + x_rad), grid_size)
    ys = np.linspace(max(y_min, best[y_idx] - y_rad), min(y_max, best[y_idx] + y_rad), grid_size)

    Z = np.empty((grid_size, grid_size), dtype=float)
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            p = best.copy()
            p[x_idx] = xv
            p[y_idx] = yv
            Z[i, j] = evaluate_individual_fitness((p, bounds_list))
    return xs, ys, Z

# -----------------------------------------------------------------------------
# 优化执行
# -----------------------------------------------------------------------------
if start_btn:
    st.session_state.stop_flag = False

    optimizer = TunedDifferentialEvolution(
        population_size=population_size,
        max_generations=max_generations,
        F_min=F_min, F_max=F_max,
        CR_min=CR_min, CR_max=CR_max,
        bounds=bounds,
        use_parallel=use_parallel,
        restart_threshold=restart_threshold,
        local_search_prob=local_search_prob,
        multi_population=multi_population
    )

    history_fitness: List[float] = []
    history_diversity: List[float] = []

    def on_step(event: Dict):
        # 图表更新
        history_fitness.append(event['best_fitness'])
        history_diversity.append(event['diversity'])
        fitness_chart.add_rows(pd.DataFrame({'best_fitness': [event['best_fitness']]}))
        diversity_chart.add_rows(pd.DataFrame({'diversity': [event['diversity']]}))

        # 元信息面板
        with meta_area:
            st.write({k: event[k] for k in ['generation','F','CR','best_fitness','diversity','stagnation_count','restart_count']})

        # 当前种群在 (t_deploy, t_fuse) 平面分布
        try:
            pop = event['population_positions']
            fit = event['population_fitness']
            df = pd.DataFrame({
                't_deploy': pop[:, 2],
                't_fuse': pop[:, 3],
                'fitness': fit
            })
            with scatter_area:
                st.scatter_chart(df, x='t_deploy', y='t_fuse', color='fitness')
        except Exception:
            pass

        # 停止控制
        if st.session_state.stop_flag:
            raise KeyboardInterrupt('Stopped by user')

    optimizer._callback = on_step

    # 启动优化
    try:
        best_position, best_fitness = optimizer.optimize()
        st.success(f'完成！最佳适应度: {best_fitness:.6f}, 最佳解: {best_position}')
    except KeyboardInterrupt:
        st.warning('已停止')
    except Exception as e:
        st.error(f'运行出错: {e}')

    # 局部最优性分析 UI
    with local_area:
        if optimizer.best_individual is None or len(history_fitness) == 0:
            st.info('请先运行一次优化以进行局部分析。')
        else:
            st.subheader('基于当前最佳解的邻域热力图')
            x_name = st.selectbox('X 轴参数', options=PARAM_NAMES, index=2)
            y_name = st.selectbox('Y 轴参数', options=PARAM_NAMES, index=3)
            x_idx = PARAM_NAMES.index(x_name)
            y_idx = PARAM_NAMES.index(y_name)
            radius_ratio = st.slider('邻域半径(相对边界比例)', 0.005, 0.25, 0.05, step=0.005)
            grid_size = st.slider('网格密度', 11, 61, 31, step=2)
            analyze = st.button('开始分析')

            if analyze:
                with st.spinner('计算热力图中...'):
                    try:
                        xs, ys, Z = compute_local_heatmap(tuple(optimizer.best_individual.position.tolist()),
                                                          optimizer.bounds_list,
                                                          x_idx, y_idx,
                                                          radius_ratio, grid_size)
                        fig, ax = plt.subplots(figsize=(6, 5))
                        im = ax.imshow(Z.T, origin='lower', aspect='auto',
                                       extent=(xs[0], xs[-1], ys[0], ys[-1]))
                        ax.set_xlabel(x_name)
                        ax.set_ylabel(y_name)
                        ax.set_title('局部适应度热力图')
                        plt.colorbar(im, ax=ax, label='fitness')
                        st.pyplot(fig)

                        # 简要局部最优性结论
                        z_center = evaluate_individual_fitness((optimizer.best_individual.position, optimizer.bounds_list))
                        z_max = float(np.max(Z))
                        z_min = float(np.min(Z))
                        if z_center >= z_max - 1e-6:
                            st.success('当前解在该二维邻域内近似局部最优。')
                        else:
                            st.warning(f'邻域内存在更优值，潜在陷入局部非最优；局部最优差距: {z_max - z_center:.4f}')
                    except Exception as e:
                        st.error(f'局部分析失败: {e}') 