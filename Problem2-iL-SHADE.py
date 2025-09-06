"""
问题2：iL-SHADE 算法求解最优无人机策略

基于 Tanabe & Fukunaga 提出的 L-SHADE/iL-SHADE：
- current-to-pbest/1 + Archive 变异
- 历史记忆自适应 F/CR（Cauchy/Normal 采样），Lehmer 加权均值更新
- 线性种群收缩（LPSR）
- 越界修复：向父代与边界的中点拉回
- 外部档案（维持多样性）

复用现有适应度接口：calculate_single_uav_single_smoke_masking
"""

import numpy as np
import time
import random
import copy
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# 求解器接口
from solver import calculate_single_uav_single_smoke_masking, TARGETS, MISSILES
from solver.trajectory import TrajectoryCalculator


@dataclass
class Individual:
    position: np.ndarray
    fitness: float


def evaluate_individual_fitness(individual_data):
    """与现有脚本一致的适应度评估接口（便于并行池调用）。"""
    position, bounds_list = individual_data
    try:
        params = {
            'v_FY1': position[0],
            'theta_FY1': position[1],
            't_deploy': position[2],
            't_fuse': position[3]
        }
        duration = calculate_single_uav_single_smoke_masking(
            uav_direction=params['theta_FY1'],
            uav_speed=params['v_FY1'],
            smoke_deploy_time=params['t_deploy'],
            smoke_explode_delay=params['t_fuse'],
            algorithm="adaptive"
        )
        return duration
    except Exception as e:
        print(f"计算错误: {e}")
        return -1000.0


def calculate_bounds() -> Dict[str, Tuple[float, float]]:
    """与精调版保持一致的搜索边界推断。"""
    traj_calc = TrajectoryCalculator()
    missile_pos = MISSILES["M1"]["initial_pos"]
    missile_speed = MISSILES["M1"]["speed"]
    fake_target = TARGETS["fake_target"]

    distance = np.sqrt(
        (fake_target[0] - missile_pos[0]) ** 2 +
        (fake_target[1] - missile_pos[1]) ** 2 +
        (fake_target[2] - missile_pos[2]) ** 2
    )
    _ = distance / missile_speed

    bounds = {
        'v_FY1': (70.0, 140.0),
        'theta_FY1': (0.0, 360.0),
        't_deploy': (0.001, 10),
        't_fuse': (0.001, 10)
    }
    return bounds


class ILShadeOptimizer:
    """iL-SHADE 优化器（单目标、连续）

    关键实现：
    - 变异：DE/current-to-pbest/1 with Archive
    - 参数自适应：M_F/M_CR 历史记忆，Cauchy/Normal 采样
    - 线性种群收缩：NP 从初始线性减小到最小
    - 档案：成功替换的父代进入档案，超限随机剔除
    """

    def __init__(self,
                 population_size_init: int = 80,
                 population_size_min: int = 20,
                 max_generations: int = 800,
                 memory_size: int = 20,
                 p_best_max: float = 0.2,
                 bounds: Optional[Dict[str, Tuple[float, float]]] = None,
                 use_parallel: bool = True,
                 archive_size_factor: float = 1.0,
                 random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)

        self.NP_init = population_size_init
        self.NP_min = max(4, population_size_min)
        self.max_generations = max_generations
        self.H = max(2, memory_size)
        self.p_best_max = min(0.5, max(0.05, p_best_max))
        self.use_parallel = use_parallel
        self.archive_size_limit = int(self.NP_init * max(0.5, archive_size_factor))

        self.bounds = calculate_bounds() if bounds is None else bounds
        self.bounds_list = list(self.bounds.values())
        self.n_dims = len(self.bounds_list)

        # 记忆向量（初始化为 0.5）
        self.M_F = np.full(self.H, 0.5, dtype=float)
        self.M_CR = np.full(self.H, 0.5, dtype=float)
        self.mem_index = 0

        # 档案（个体位置组成的列表）
        self.archive: List[np.ndarray] = []

        # 状态
        self.population: List[Individual] = []
        self.best: Optional[Individual] = None
        self.fitness_history: List[float] = []
        self.NP = self.NP_init

        # 并行设置
        if self.use_parallel:
            self.n_processes = min(mp.cpu_count(), max(2, self.NP_init // 2))

    # --- 基础工具 ---
    def _sample_F(self, M_F_k: float) -> float:
        # Cauchy(M_F_k, 0.1)，截断到 (0,1]，若<=0 反复重采样；避免过小步长
        while True:
            F = np.random.standard_cauchy() * 0.1 + M_F_k
            if F > 0:
                break
        F = float(min(F, 1.0))
        if F < 1e-3:
            F = 1e-3
        return F

    def _sample_CR(self, M_CR_k: float) -> float:
        # Normal(M_CR_k, 0.1)，截断到 [0,1]
        CR = np.random.normal(loc=M_CR_k, scale=0.1)
        return float(np.clip(CR, 0.0, 1.0))

    def _repair_bounds(self, trial: np.ndarray, target: np.ndarray) -> np.ndarray:
        # 越界修复：与边界中点靠近父代
        for j in range(self.n_dims):
            lb, ub = self.bounds_list[j]
            if trial[j] < lb:
                trial[j] = (target[j] + lb) / 2.0
            elif trial[j] > ub:
                trial[j] = (target[j] + ub) / 2.0
        return trial

    def _initialize_population(self):
        self.population = []
        for _ in range(self.NP_init):
            position = np.array([
                np.random.uniform(lb, ub) for (lb, ub) in self.bounds_list
            ])
            self.population.append(Individual(position=position, fitness=-np.inf))
        self._evaluate_population()
        self._update_best()

    def _evaluate_population(self):
        if self.use_parallel:
            data = [(ind.position, self.bounds_list) for ind in self.population]
            with ProcessPoolExecutor(max_workers=self.n_processes) as ex:
                futures = {ex.submit(evaluate_individual_fitness, d): i for i, d in enumerate(data)}
                for f in as_completed(futures):
                    idx = futures[f]
                    self.population[idx].fitness = f.result()
        else:
            for ind in self.population:
                ind.fitness = evaluate_individual_fitness((ind.position, self.bounds_list))

    def _update_best(self):
        for ind in self.population:
            if self.best is None or ind.fitness > self.best.fitness:
                self.best = copy.deepcopy(ind)

    def _linear_pop_size_reduction(self, gen: int):
        # NP(g) = round(NP_min + (NP_init - NP_min) * (1 - gen/max_gen))
        target_np = int(round(self.NP_min + (self.NP_init - self.NP_min) * (1.0 - gen / max(1, self.max_generations))))
        if target_np < self.NP:
            # 按适应度排序，保留前 target_np
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            self.population = self.population[:target_np]
            self.NP = target_np
            # 修正档案上限
            while len(self.archive) > max(1, self.NP):
                pop_idx = np.random.randint(0, len(self.archive))
                self.archive.pop(pop_idx)

    def _select_pbest_index(self, p_fraction: float) -> int:
        num_top = max(1, int(np.ceil(p_fraction * self.NP)))
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        return np.random.randint(0, num_top)

    def _mutation_current_to_pbest_1(self, i: int, F: float, p_fraction: float) -> np.ndarray:
        # pbest index in the sorted population
        pbest_idx = self._select_pbest_index(p_fraction)
        # r1 from population excluding i and pbest
        candidates_r1 = [idx for idx in range(self.NP) if idx != i and idx != pbest_idx]
        r1 = random.choice(candidates_r1)
        # r2 from population + archive excluding {i, r1, pbest}
        union = list(range(self.NP))
        union_ex = set([i, r1, pbest_idx])
        pool = [idx for idx in union if idx not in union_ex]
        choose_archive = False
        if len(self.archive) > 0:
            # 以 50% 概率从档案取一份以增强多样性
            choose_archive = random.random() < 0.5
        if choose_archive:
            x_r2 = random.choice(self.archive)
        else:
            r2 = random.choice(pool)
            x_r2 = self.population[r2].position

        x_i = self.population[i].position
        x_pbest = self.population[pbest_idx].position
        x_r1 = self.population[r1].position

        mutant = x_i + F * (x_pbest - x_i) + F * (x_r1 - x_r2)
        return mutant

    def _binomial_crossover(self, target: np.ndarray, mutant: np.ndarray, CR: float) -> np.ndarray:
        trial = target.copy()
        j_rand = np.random.randint(0, self.n_dims)
        for j in range(self.n_dims):
            if np.random.random() < CR or j == j_rand:
                trial[j] = mutant[j]
        return self._repair_bounds(trial, target)

    def _update_memory(self, S_F: List[float], S_CR: List[float], weights: List[float]):
        if len(S_F) == 0:
            return
        w = np.array(weights, dtype=float)
        w = w / (np.sum(w) + 1e-12)
        S_F = np.array(S_F, dtype=float)
        S_CR = np.array(S_CR, dtype=float)

        # Lehmer 均值（加权）：sum(w*F^2) / sum(w*F)
        MF_new = np.sum(w * (S_F ** 2)) / max(1e-12, np.sum(w * S_F))
        # 加权算术均值
        MCR_new = np.sum(w * S_CR)

        self.M_F[self.mem_index] = float(np.clip(MF_new, 1e-8, 1.0))
        self.M_CR[self.mem_index] = float(np.clip(MCR_new, 0.0, 1.0))
        self.mem_index = (self.mem_index + 1) % self.H

    def _shrink_archive(self):
        limit = max(1, int(self.archive_size_limit))
        while len(self.archive) > limit:
            self.archive.pop(random.randrange(len(self.archive)))

    def optimize(self):
        print("=" * 60)
        print("开始 iL-SHADE 优化")
        print("=" * 60)

        self._initialize_population()

        for gen in range(self.max_generations):
            # 线性种群收缩
            self._linear_pop_size_reduction(gen)

            # 逐个体生成试验向量
            new_population: List[Individual] = []
            S_F, S_CR, weights = [], [], []

            # 采样对应的记忆槽
            mem_indices = np.random.randint(0, self.H, size=self.NP)

            for i in range(self.NP):
                k = mem_indices[i]
                F_i = self._sample_F(self.M_F[k])
                CR_i = self._sample_CR(self.M_CR[k])
                # p 随机均匀取 [pmin, p_best_max]
                pmin = max(2.0 / max(2, self.NP), 0.02)
                p_i = np.random.uniform(pmin, self.p_best_max)

                mutant = self._mutation_current_to_pbest_1(i, F_i, p_i)
                trial = self._binomial_crossover(self.population[i].position, mutant, CR_i)

                # 评估
                f_trial = evaluate_individual_fitness((trial, self.bounds_list))
                f_target = self.population[i].fitness

                if f_trial > f_target:
                    # 成功：trial 取代 target；target 进入档案
                    new_population.append(Individual(position=trial, fitness=f_trial))
                    self.archive.append(self.population[i].position.copy())
                    # 记录成功参数及权重（按提升量加权）
                    S_F.append(F_i)
                    S_CR.append(CR_i)
                    weights.append(max(1e-12, f_trial - f_target))
                else:
                    new_population.append(copy.deepcopy(self.population[i]))

            # 更新记忆
            self._update_memory(S_F, S_CR, weights)

            # 修剪档案
            self._shrink_archive()

            # 更新种群与最好值
            self.population = new_population
            self._update_best()
            self.fitness_history.append(self.best.fitness)

            print(f"第 {gen + 1}/{self.max_generations} 代 | 最佳: {self.best.fitness:.6f} | NP: {self.NP}")

            # 目标阈值（与精调版保持一致）
            if self.best.fitness >= 4.85:
                print("  🏆 发现超优解！")
                break
            elif self.best.fitness >= 4.79:
                print("  🎯 达到目标解！继续稳态搜索……")

        return self.best.position, self.best.fitness


def main():
    print("问题2：iL-SHADE 优化器")
    optimizer = ILShadeOptimizer(
        population_size_init=80,
        population_size_min=20,
        max_generations=800,
        memory_size=20,
        p_best_max=0.2,
        use_parallel=True,
        archive_size_factor=1.0,
        random_seed=None,
    )
    start = time.time()
    best_pos, best_fit = optimizer.optimize()
    end = time.time()

    print("-" * 40)
    print(f"最佳适应度: {best_fit:.6f}")
    print(f"耗时: {end - start:.2f}s")
    print(f"最佳参数: v={best_pos[0]:.3f}, theta={best_pos[1]:.3f}, t_deploy={best_pos[2]:.3f}, t_fuse={best_pos[3]:.3f}")
    return {
        'best_position': best_pos,
        'best_fitness': best_fit,
        'fitness_history': optimizer.fitness_history,
    }


if __name__ == "__main__":
    _ = main() 