import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import logging
from concurrent.futures import ProcessPoolExecutor
from numba import njit

# ================= 日志配置 =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ================= Numba 加速函数 =================
@njit(fastmath=True)
def dist_point_to_line_segment(p, a, b):
    ap = p - a
    ab = b - a
    ab_squared = np.dot(ab, ab)
    if ab_squared == 0.0:
        return np.linalg.norm(ap)
    t = np.dot(ap, ab) / ab_squared
    if t < 0.0:
        return np.linalg.norm(p - a)
    elif t > 1.0:
        return np.linalg.norm(p - b)
    projection = a + t * ab
    return np.linalg.norm(p - projection)


class DroneObscurationOptimizer:
    def __init__(self):
        # --- 基本物理参数 ---
        self.G = 9.8
        self.MISSILE_V = 300.0
        self.CLOUD_SINK_V = 3.0
        self.CLOUD_RADIUS = 10.0
        self.CLOUD_LIFETIME = 20.0

        # --- 坐标 ---
        self.M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
        self.FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])
        self.TRUE_TARGET_CENTER_POS = np.array([0.0, 200.0, 5.0])
        self.DRONE_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])
        self.BOMB_NAMES = ['1号烟幕弹', '2号烟幕弹', '3号烟幕弹']

        # --- 导弹轨迹缓存 ---
        missile_path_vector = self.FALSE_TARGET_POS - self.M1_INITIAL_POS
        self.missile_path_dist = np.linalg.norm(missile_path_vector)
        self.missile_unit_vector = missile_path_vector / self.missile_path_dist
        self.missile_total_flight_time = self.missile_path_dist / self.MISSILE_V

        # 预缓存导弹轨迹
        self.dt = 0.05
        self.time_grid = np.arange(0, self.missile_total_flight_time + self.dt, self.dt)
        self.missile_positions = (
            self.M1_INITIAL_POS[None, :] + self.missile_unit_vector[None, :] * self.MISSILE_V * self.time_grid[:, None]
        )

        # --- 参数边界 ---
        self.DRONE_BOUNDS = [
            (70, 140),
            (0, 2 * np.pi),
            (1, self.missile_total_flight_time / 3),
            (1, 20),
            (self.missile_total_flight_time / 3 + 1, 2 * self.missile_total_flight_time / 3),
            (1, 20),
            (2 * self.missile_total_flight_time / 3 + 1, self.missile_total_flight_time - 5),
            (1, 20),
        ]

        # 存档点目录
        self.checkpoint_dir = os.path.join(os.path.dirname(__file__), 'Data', 'Q3')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _calculate_obscuration_intervals(self, params_8d):
        obscuration_times = set()
        v, theta, t_fly1, t_fall1, t_fly2, t_fall2, t_fly3, t_fall3 = params_8d

        if not (t_fly1 + 1 <= t_fly2 and t_fly2 + 1 <= t_fly3):
            return -1000

        bomb_params = [(t_fly1, t_fall1), (t_fly2, t_fall2), (t_fly3, t_fall3)]
        drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])

        for (t_fly, t_fall) in bomb_params:
            drop_pos = self.DRONE_INITIAL_POS + drone_v_vector * t_fly
            fall_time_max = math.sqrt(2 * drop_pos[2] / self.G)
            if t_fall > fall_time_max:
                continue

            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            if detonation_pos_z < 0:
                continue

            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            detonation_time = t_fly + t_fall

            start_idx = int(detonation_time / self.dt)
            end_idx = int((detonation_time + self.CLOUD_LIFETIME) / self.dt)
            end_idx = min(end_idx, len(self.time_grid))

            for idx in range(start_idx, end_idx):
                t = self.time_grid[idx]
                missile_pos = self.missile_positions[idx]
                cloud_center = detonation_pos - np.array([0, 0, self.CLOUD_SINK_V * (t - detonation_time)])
                distance = dist_point_to_line_segment(cloud_center, missile_pos, self.TRUE_TARGET_CENTER_POS)
                if distance <= self.CLOUD_RADIUS:
                    obscuration_times.add(idx)

        return len(obscuration_times) * self.dt

    def _objective_function(self, x):
        return -self._calculate_obscuration_intervals(x)

    def genetic_algorithm(self, population_size=50, max_generations=500, checkpoint_interval=50, init_population=None, start_generation=0):
        dim = len(self.DRONE_BOUNDS)
        if init_population is None:
            population = np.zeros((population_size, dim))
            for i in range(dim):
                low, high = self.DRONE_BOUNDS[i]
                population[:, i] = np.random.uniform(low, high, population_size)
        else:
            population = init_population

        best_fitness = float('-inf')
        best_individual = None
        pbar = tqdm(total=max_generations, desc="优化进度")

        for generation in range(start_generation, max_generations):
            # 并行计算适应度
            with ProcessPoolExecutor() as executor:
                fitness = list(executor.map(self._objective_function, population))
            fitness = np.array(fitness)

            max_fitness = -np.min(fitness)
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = population[np.argmin(fitness)].copy()
                logging.info(f"Generation {generation}: Best obscuration time = {best_fitness:.2f} s")

            # 存档点
            if (generation + 1) % checkpoint_interval == 0:
                checkpoint_data = {
                    'generation': generation + 1,
                    'best_params': best_individual,
                    'population': population
                }
                path = os.path.join(self.checkpoint_dir, 'checkpoint.npy')
                np.save(path, checkpoint_data)
                logging.info(f"检查点已保存: {path}")

            # 锦标赛选择
            new_population = []
            tournament_size = 5
            for _ in range(population_size):
                idxs = np.random.choice(population_size, tournament_size, replace=False)
                winner_idx = idxs[np.argmin(fitness[idxs])]
                new_population.append(population[winner_idx].copy())

            # 交叉
            for i in range(0, population_size, 2):
                if i + 1 < population_size and np.random.random() < 0.8:
                    p1, p2 = new_population[i], new_population[i + 1]
                    mask = np.random.random(dim) < 0.5
                    c1 = np.where(mask, p1, p2)
                    c2 = np.where(mask, p2, p1)
                    new_population[i], new_population[i + 1] = c1, c2

            # 变异
            for i in range(population_size):
                if np.random.random() < 0.1:
                    for j in range(dim):
                        low, high = self.DRONE_BOUNDS[j]
                        new_population[i][j] += np.random.normal(0, (high - low) / 10)
                        new_population[i][j] = np.clip(new_population[i][j], low, high)

            population = np.array(new_population)
            pbar.update(1)

        pbar.close()
        logging.info(f"优化完成，最佳遮蔽时间: {best_fitness:.2f} 秒")
        return best_individual

    def resume_from_checkpoint(self, population_size=50, max_generations=500, checkpoint_interval=50):
        path = os.path.join(self.checkpoint_dir, 'checkpoint.npy')
        if not os.path.exists(path):
            logging.error("没有找到检查点，无法恢复。")
            return self.genetic_algorithm(population_size, max_generations, checkpoint_interval)

        checkpoint_data = np.load(path, allow_pickle=True).item()
        start_generation = checkpoint_data['generation']
        best_params = checkpoint_data['best_params']
        population = checkpoint_data['population']

        logging.info(f"从第 {start_generation} 代恢复优化...")
        return self.genetic_algorithm(
            population_size=population_size,
            max_generations=max_generations,
            checkpoint_interval=checkpoint_interval,
            init_population=population,
            start_generation=start_generation
        )


# ================= 主程序 =================
if __name__ == '__main__':
    optimizer = DroneObscurationOptimizer()
    POP_SIZE = 50
    MAX_GENERATIONS = 200
    CHECKPOINT_INTERVAL = 20

    checkpoint_path = os.path.join(optimizer.checkpoint_dir, 'checkpoint.npy')
    if os.path.exists(checkpoint_path):
        logging.info("检测到检查点，将恢复优化...")
        best = optimizer.resume_from_checkpoint(
            population_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
    else:
        logging.info("未检测到检查点，将开始新的优化...")
        best = optimizer.genetic_algorithm(
            population_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )

    logging.info(f"最终最优参数: {best}")
