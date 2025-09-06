"""
问题4：Primitive 版本 - 三架无人机（FY1, FY2, FY3）分别优化与遮蔽区间甘特图

整体框架参考 Problem2-DE-Tuned.py，但做如下调整：
- 依次分别优化 FY1、FY2、FY3 的单烟幕弹参数，代数固定为 100
- 输出每架无人机的最佳参数、遮蔽总时长与遮蔽区间列表
- 使用甘特图绘制三架无人机各自的遮蔽区间
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from solver.core import MaskingCalculator, find_t_intervals, find_t_intervals_adaptive, find_t_intervals_smart
from solver import TARGETS, MISSILES
from solver.config import SMOKE_PARAMS

# 进程内全局缓存，避免重复构造重对象
_GLOBAL = {"calc": None, "missile_traj": None, "tmax": None}

def _get_worker_state():
    if _GLOBAL["calc"] is None:
        calc = MaskingCalculator()
        mtraj = calc.traj_calc.create_missile_trajectory("M1")
        _, tmax = calc.traj_calc.get_trajectory_bounds(mtraj, calc.max_time)
        _GLOBAL.update(calc=calc, missile_traj=mtraj, tmax=tmax)
    return _GLOBAL["calc"], _GLOBAL["missile_traj"], _GLOBAL["tmax"]


@dataclass
class Individual:
    position: np.ndarray
    fitness: float


def calculate_bounds() -> Dict[str, Tuple[float, float]]:
    """与问题2保持一致的参数边界，扩大时间上界至 66。"""
    bounds = {
        'v_FY': (70.0, 140.0),
        'theta_FY': (0.0, 360.0),
        't_deploy': (0.001, 66.0),
        't_fuse': (0.001, 66.0),
    }
    return bounds


def compute_masking_intervals_for_fy(
    fy_id: str,
    uav_direction: float,
    uav_speed: float,
    smoke_deploy_time: float,
    smoke_explode_delay: float,
    algorithm: str = "adaptive"
) -> Tuple[List[Tuple[float, float]], float]:
    """计算指定 FY 的遮蔽区间与总时长。"""
    calc = MaskingCalculator()

    missile_traj = calc.traj_calc.create_missile_trajectory("M1")
    uav_traj = calc.traj_calc.create_uav_trajectory(fy_id, direction_degrees=uav_direction, speed=uav_speed)
    smoke_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, smoke_deploy_time, smoke_explode_delay)

    predicate = calc._create_masking_predicate(missile_traj, smoke_traj)

    explode_time = smoke_deploy_time + smoke_explode_delay
    _, global_end = calc.traj_calc.get_trajectory_bounds(missile_traj, calc.max_time)
    end_time = global_end  # 结果展示使用完整时间窗口，避免上界被裁剪

    if algorithm == "fixed":
        intervals = find_t_intervals(predicate, calc.threshold, explode_time, end_time, calc.time_step)
    elif algorithm == "adaptive":
        intervals = find_t_intervals_adaptive(
            predicate, calc.threshold, explode_time, end_time,
            initial_step=calc.time_step * 10,
            min_step=calc.time_step / 2,
            max_step=calc.time_step * 50
        )
    elif algorithm == "smart":
        intervals = find_t_intervals_smart(
            predicate, calc.threshold, explode_time, end_time,
            initial_step=calc.time_step * 5,
            aggressive_speedup=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    total_duration = float(sum(b - a for a, b in intervals))
    return intervals, total_duration


def _evaluate_individual_fitness_for_fy(args) -> float:
    """并行池调用的适应度评估函数（带进程内缓存与窗口裁剪）。"""
    position, fy_id = args
    try:
        calc, missile_traj, tmax = _get_worker_state()
        uav_speed = float(position[0])
        uav_theta = float(position[1])
        t_deploy = float(position[2])
        t_fuse = float(position[3])

        uav_traj = calc.traj_calc.create_uav_trajectory(fy_id, direction_degrees=uav_theta, speed=uav_speed)
        smoke_traj = calc.traj_calc.create_smoke_trajectory(uav_traj, t_deploy, t_fuse)
        predicate = calc._create_masking_predicate(missile_traj, smoke_traj)

        explode_time = t_deploy + t_fuse
        # 评估阶段仍使用裁剪窗口以提速；但最终展示用完整窗口
        end_time = min(tmax, explode_time + float(SMOKE_PARAMS.get("duration", 20)) + 5.0)

        intervals = find_t_intervals_adaptive(
            predicate, calc.threshold, explode_time, end_time,
            initial_step=0.5,
            min_step=calc.time_step,
            max_step=2.0
        )
        duration = float(sum(b - a for a, b in intervals))
        return duration
    except Exception as e:
        print(f"计算错误[{fy_id}]: {e}")
        return -1000.0


class PrimitiveDE:
    """简化的差分进化（DE/rand/1/bin），用于 100 代快速优化。"""

    def __init__(self,
                 fy_id: str,
                 population_size: int = 60,
                 max_generations: int = 100,
                 F: float = 0.7,
                 CR: float = 0.9,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_parallel: bool = True):
        self.fy_id = fy_id
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.bounds = calculate_bounds() if bounds is None else bounds
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)
        self.use_parallel = use_parallel

        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), max(2, self.population_size // 2))
            # 持久化进程池
            self._executor = ProcessPoolExecutor(max_workers=self.n_processes)
        else:
            self._executor = None

        self.population: List[Individual] = []
        self.best: Optional[Individual] = None
        self.fitness_history: List[float] = []

    def _init_population(self):
        self.population = []
        for _ in range(self.population_size):
            position = np.array([
                np.random.uniform(lb, ub) for lb, ub in self.bounds_list
            ])
            self.population.append(Individual(position=position, fitness=-np.inf))
        self._evaluate_population()
        self._update_best()

    def _evaluate_population(self):
        if self.use_parallel:
            data = [(ind.position, self.fy_id) for ind in self.population]
            futures = {self._executor.submit(_evaluate_individual_fitness_for_fy, d): i for i, d in enumerate(data)}
            for f in as_completed(futures):
                idx = futures[f]
                self.population[idx].fitness = float(f.result())
        else:
            for ind in self.population:
                ind.fitness = float(_evaluate_individual_fitness_for_fy((ind.position, self.fy_id)))

    def _update_best(self):
        for ind in self.population:
            if self.best is None or ind.fitness > self.best.fitness:
                self.best = copy.deepcopy(ind)

    def _mutate(self, target_idx: int) -> np.ndarray:
        idxs = list(range(self.population_size))
        idxs.remove(target_idx)
        r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
        mutant = self.population[r1].position + self.F * (self.population[r2].position - self.population[r3].position)
        # 边界裁剪
        for j in range(self.n_dims):
            lb, ub = self.bounds_list[j]
            mutant[j] = np.clip(mutant[j], lb, ub)
        return mutant

    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        trial = target.copy()
        j_rand = np.random.randint(0, self.n_dims)
        for j in range(self.n_dims):
            if np.random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        return trial

    def optimize(self) -> Tuple[np.ndarray, float]:
        print(f"=== {self.fy_id} 优化开始：DE/rand/1/bin，代数={self.max_generations} ===")
        self._init_population()
        for gen in range(self.max_generations):
            new_population: List[Individual] = []
            trials: List[np.ndarray] = []
            for i in range(self.population_size):
                mutant = self._mutate(i)
                trial = self._crossover(self.population[i].position, mutant)
                # 边界裁剪
                for j in range(self.n_dims):
                    lb, ub = self.bounds_list[j]
                    trial[j] = np.clip(trial[j], lb, ub)
                trials.append(trial)
            # 并行评估试验向量（复用持久进程池）
            if self.use_parallel:
                data = [(trials[i], self.fy_id) for i in range(self.population_size)]
                futures = {self._executor.submit(_evaluate_individual_fitness_for_fy, d): i for i, d in enumerate(data)}
                f_trial_list = [None] * self.population_size
                for f in as_completed(futures):
                    idx = futures[f]
                    f_trial_list[idx] = float(f.result())
            else:
                f_trial_list = [float(_evaluate_individual_fitness_for_fy((trials[i], self.fy_id))) for i in range(self.population_size)]
            # 选择
            for i in range(self.population_size):
                f_trial = f_trial_list[i]
                if f_trial > self.population[i].fitness:
                    new_population.append(Individual(position=trials[i], fitness=f_trial))
                else:
                    new_population.append(copy.deepcopy(self.population[i]))
            self.population = new_population
            self._update_best()
            self.fitness_history.append(self.best.fitness)
            if (gen + 1) % 5 == 0 or gen == 0:
                print(f"[{self.fy_id}] 代 {gen+1:03d}/{self.max_generations} | 最佳={self.best.fitness:.6f}")
        print(f"=== {self.fy_id} 优化结束，最佳={self.best.fitness:.6f} ===")
        # 关闭进程池
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        return self.best.position, self.best.fitness


def plot_gantt(intervals_map: Dict[str, List[Tuple[float, float]]]):
    """绘制三架无人机遮蔽区间甘特图。"""
    fig, ax = plt.subplots(figsize=(10, 3.6))
    colors = {"FY1": "tab:blue", "FY2": "tab:orange", "FY3": "tab:green"}
    y_positions = {"FY1": 30, "FY2": 20, "FY3": 10}
    height = 6

    for fy_id, intervals in intervals_map.items():
        bars = [(a, b - a) for a, b in intervals]
        if len(bars) > 0:
            ax.broken_barh(bars, (y_positions[fy_id], height), facecolors=colors[fy_id], alpha=0.8, label=fy_id)
        else:
            # 空区间占位（不画）
            pass

    ax.set_xlabel("时间 (s)")
    ax.set_yticks([y_positions["FY3"], y_positions["FY2"], y_positions["FY1"]])
    ax.set_yticklabels(["FY3", "FY2", "FY1"])
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.set_title("三架无人机遮蔽区间甘特图")

    # 去重图例
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if len(uniq) > 0:
        ax.legend(uniq.values(), uniq.keys(), loc='upper right')

    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(42)

    bounds = calculate_bounds()

    results: Dict[str, Dict[str, object]] = {}

    # 仅优化 FY3（专注收敛到可行解）
    for fy in ["FY1","FY2", "FY3"]:
        optimizer = PrimitiveDE(
            fy_id=fy,
            population_size=80,
            max_generations=400,
            F=0.7,
            CR=0.9,
            bounds=bounds,
            use_parallel=True,
        )
        start = time.time()
        best_pos, best_fit = optimizer.optimize()
        end = time.time()

        # 提取区间用于甘特图
        intervals, total = compute_masking_intervals_for_fy(
            fy_id=fy,
            uav_direction=float(best_pos[1]),
            uav_speed=float(best_pos[0]),
            smoke_deploy_time=float(best_pos[2]),
            smoke_explode_delay=float(best_pos[3]),
            algorithm="adaptive",
        )

        results[fy] = {
            "best_position": best_pos,
            "best_fitness": best_fit,
            "intervals": intervals,
            "time": end - start,
        }

        print(f"\n[{fy}] 结果：")
        print(f"  最佳适应度（总遮蔽时长）: {best_fit:.6f} s")
        print(f"  最佳参数: v={best_pos[0]:.3f}, theta={best_pos[1]:.3f}, t_deploy={best_pos[2]:.3f}, t_fuse={best_pos[3]:.3f}")
        print(f"  遮蔽区间数量: {len(intervals)}")
        if len(intervals) > 0:
            print(f"  示例区间(前5个): {intervals[:5]}")
        print(f"  优化耗时: {end - start:.2f} s")

    # 绘制甘特图（仅 FY3）
    intervals_map = {fy: results[fy]["intervals"] for fy in ["FY1", "FY2", "FY3"]}
    plot_gantt(intervals_map)

    return results


if __name__ == "__main__":
    _ = main()
