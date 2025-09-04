import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import math
import datetime
import logging

# 依赖：numpy, pandas, tqdm, openpyxl
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DroneObscurationOptimizer:
    """
    使用遗传算法解决无人机协同烟幕干扰问题的优化器，
    针对导弹 M1 和无人机 FY1，优化遮蔽时间。
    """

    def __init__(self):
        # --- 基本物理和场景常量 ---
        self.G = 9.8  # 重力加速度
        self.MISSILE_V = 300.0  # 导弹速度
        self.CLOUD_SINK_V = 3.0  # 烟云下沉速度
        self.CLOUD_RADIUS = 10.0  # 烟云半径（题目要求中心10m范围内有效）
        self.CLOUD_LIFETIME = 20.0  # 烟云持续时间
        if any(x <= 0 for x in [self.G, self.MISSILE_V, self.CLOUD_SINK_V, self.CLOUD_RADIUS, self.CLOUD_LIFETIME]):
            raise ValueError("物理常量必须为正值")
        
        # --- 坐标定义 ---
        self.M1_INITIAL_POS = np.array([20000.0, 0.0, 2000.0])
        self.FALSE_TARGET_POS = np.array([0.0, 0.0, 0.0])
        self.TRUE_TARGET_CENTER_POS = np.array([0.0, 200.0, 5.0])  # 圆柱形目标中心
        self.DRONE_INITIAL_POS = np.array([17800.0, 0.0, 1800.0])  # FY1
        self.BOMB_NAMES = ['1号烟幕弹', '2号烟幕弹', '3号烟幕弹']

        # --- 导弹路径预计算 ---
        missile_path_vector = self.FALSE_TARGET_POS - self.M1_INITIAL_POS
        self.missile_path_dist = np.linalg.norm(missile_path_vector)
        self.missile_unit_vector = missile_path_vector / self.missile_path_dist  # 添加单位向量初始化
        self.missile_total_flight_time = self._calculate_missile_flight_time()

        # 优化参数边界: [速度, 航向(弧度), 3个投放时间, 3个自由落体时间]
        self.DRONE_BOUNDS = [
            (70, 140),          # 无人机速度
            (0, 2 * np.pi),     # 航向 (0~360°，转为弧度)
            (1, self.missile_total_flight_time / 3),  # 第1枚投放时间
            (1, 20),            # 第1枚自由落体时间
            (self.missile_total_flight_time / 3 + 1, 2 * self.missile_total_flight_time / 3),  # 第2枚
            (1, 20),
            (2 * self.missile_total_flight_time / 3 + 1, self.missile_total_flight_time - 5),  # 第3枚
            (1, 20)
        ]

    def _calculate_missile_flight_time(self):
        """计算导弹总飞行时间"""
        missile_path_vector = self.FALSE_TARGET_POS - self.M1_INITIAL_POS
        missile_path_dist = np.linalg.norm(missile_path_vector)
        return missile_path_dist / self.MISSILE_V

    def _get_missile_pos(self, t):
        """计算导弹在时间 t 的位置"""
        if t > self.missile_total_flight_time:
            return self.FALSE_TARGET_POS
        return self.M1_INITIAL_POS + self.missile_unit_vector * self.MISSILE_V * t

    def _dist_point_to_line_segment(self, p, a, b):
        """计算点 p 到线段 (a, b) 的最短距离"""
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

    def _calculate_obscuration_intervals(self, params_8d):
        """计算给定8个参数下的总遮蔽时间"""
        obscuration_times = set()
        v, theta, t_fly1, t_fall1, t_fly2, t_fall2, t_fly3, t_fall3 = params_8d
        
        # 确保投放时间顺序和间隔约束
        if not (t_fly1 + 1 <= t_fly2 and t_fly2 + 1 <= t_fly3):
            logging.debug(f"Invalid time order: t_fly1={t_fly1:.2f}, t_fly2={t_fly2:.2f}, t_fly3={t_fly3:.2f}")
            return -1000
        
        bomb_params = [(t_fly1, t_fall1), (t_fly2, t_fall2), (t_fly3, t_fall3)]
        drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])

        for i, (t_fly, t_fall) in enumerate(bomb_params):
            drop_pos = self.DRONE_INITIAL_POS + drone_v_vector * t_fly
            fall_time_max = math.sqrt(2 * drop_pos[2] / self.G)
            if t_fall > fall_time_max:
                logging.debug(f"Bomb {i+1} fall time too long: t_fall={t_fall:.2f}, max={fall_time_max:.2f}")
                continue
            
            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            
            if detonation_pos[2] < 0:
                logging.debug(f"Bomb {i+1} detonation below ground: z={detonation_pos[2]:.2f}")
                continue
            
            detonation_time = t_fly + t_fall
            start_time = detonation_time
            end_time = detonation_time + self.CLOUD_LIFETIME
            
            for t in np.arange(start_time, end_time, 0.05):
                if t > self.missile_total_flight_time:
                    break
                missile_pos = self._get_missile_pos(t)
                cloud_center = detonation_pos - np.array([0, 0, self.CLOUD_SINK_V * (t - detonation_time)])
                distance = self._dist_point_to_line_segment(p=cloud_center, a=missile_pos, b=self.TRUE_TARGET_CENTER_POS)
                
                logging.debug(f"Bomb {i+1}, t={t:.2f}, missile_pos={missile_pos}, cloud_center={cloud_center}, distance={distance:.2f}")
                if distance <= self.CLOUD_RADIUS:
                    obscuration_times.add(round(t, 2))
        
        return len(obscuration_times) * 0.05

    def _objective_function(self, x):
        """遗传算法的适应度函数，返回负遮蔽时间（最大化问题转为最小化）"""
        return -self._calculate_obscuration_intervals(x)

    def _format_and_save_output(self, best_params, prefix=""):
        """将结果格式化为 DataFrame 并保存到 Excel 文件"""
        v, theta, t_fly1, t_fall1, t_fly2, t_fall2, t_fly3, t_fall3 = best_params
        bomb_params = [(t_fly1, t_fall1), (t_fly2, t_fall2), (t_fly3, t_fall3)]
        output_data = []
        
        drone_v_vector = np.array([v * np.cos(theta), v * np.sin(theta), 0])

        for i, (t_fly, t_fall) in enumerate(bomb_params):
            drop_pos = self.DRONE_INITIAL_POS + drone_v_vector * t_fly
            detonation_pos_z = drop_pos[2] - 0.5 * self.G * t_fall**2
            detonation_pos_xy = drop_pos[:2] + drone_v_vector[:2] * t_fall
            detonation_pos = np.array([detonation_pos_xy[0], detonation_pos_xy[1], detonation_pos_z])
            
            row = {
                '无人机编号': 'FY1',
                '烟幕弹编号': self.BOMB_NAMES[i],
                '无人机运动方向 (度)': math.degrees(theta) % 360,
                '无人机运动速度 (m/s)': v,
                '烟幕干扰弹投放点的x坐标 (m)': drop_pos[0],
                '烟幕干扰弹投放点的y坐标 (m)': drop_pos[1],
                '烟幕干扰弹投放点的z坐标 (m)': drop_pos[2],
                '烟幕干扰弹起爆点的x坐标 (m)': detonation_pos[0],
                '烟幕干扰弹起爆点的y坐标 (m)': detonation_pos[1],
                '烟幕干扰弹起爆点的z坐标 (m)': detonation_pos[2],
            }
            output_data.append(row)
            
        df_details = pd.DataFrame(output_data)
        best_obscuration_time = -self._objective_function(best_params)
        df_summary = pd.DataFrame([{'总有效遮蔽时长 (s)': best_obscuration_time}])

        output_dir = 'Data/Q3'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, f'{prefix}result1.xlsx')

        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df_details.to_excel(writer, sheet_name='策略详情', index=False)
                df_summary.to_excel(writer, sheet_name='总结', index=False)
            logging.info(f"结果已成功保存到: {output_path}")
        except Exception as e:
            logging.error(f"保存Excel文件失败: {e}")
        return df_details

    def genetic_algorithm(self, population_size=50, max_generations=500, 
                         checkpoint_interval=50, init_population=None):
        """使用遗传算法优化无人机投放策略"""
        if population_size < 5:
            logging.warning("种群大小必须大于4，已自动设置为5。")
            population_size = 5
            
        logging.info(f"开始进行遗传算法优化...")
        logging.info(f"优化参数: 种群大小={population_size}, 最大代数={max_generations}")
        
        # 初始化种群
        dim = len(self.DRONE_BOUNDS)
        if init_population is None:
            population = np.zeros((population_size, dim))
            for i in range(dim):
                low, high = self.DRONE_BOUNDS[i]
                population[:, i] = np.random.uniform(low, high, population_size)
            # 启发式初始化第一个个体
            population[0] = [
                100,  # 中间速度
                np.pi / 4,  # 45°
                self.missile_total_flight_time / 4,
                5,
                self.missile_total_flight_time / 2,
                5,
                3 * self.missile_total_flight_time / 4,
                5
            ]
        else:
            population = init_population
        
        pbar = tqdm(total=max_generations, desc="优化进度")
        best_fitness = float('-inf')
        best_individual = None
        
        for generation in range(max_generations):
            # 计算适应度
            fitness = np.array([self._objective_function(ind) for ind in population])
            max_fitness = -np.min(fitness)  # 负值转为正遮蔽时间
            if max_fitness > best_fitness:
                best_fitness = max_fitness
                best_individual = population[np.argmin(fitness)].copy()
                logging.info(f"Generation {generation}: Best obscuration time = {best_fitness:.2f} s")
            
            # 保存检查点
            if (generation + 1) % checkpoint_interval == 0:
                logging.info(f"Generation {generation + 1}: Saving checkpoint...")
                checkpoint_data = {'generation': generation + 1, 'best_params': best_individual}
                checkpoint_dir = 'Data/Q3'
                os.makedirs(checkpoint_dir, exist_ok=True)
                np.save(os.path.join(checkpoint_dir, 'checkpoint.npy'), checkpoint_data)
                logging.info(f"检查点已保存到: {os.path.join(checkpoint_dir, 'checkpoint.npy')}")
                self._format_and_save_output(best_individual, prefix=f'iter_{generation + 1}_')
                pbar.set_description(f"优化进度 (检查点已保存)")

            # 锦标赛选择
            new_population = []
            tournament_size = 5
            for _ in range(population_size):
                tournament_indices = np.random.choice(population_size, tournament_size, replace=False)
                tournament_fitness = fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                new_population.append(population[winner_idx].copy())
            
            # 交叉
            for i in range(0, population_size, 2):
                if i + 1 < population_size and np.random.random() < 0.8:  # 交叉概率
                    parent1, parent2 = new_population[i], new_population[i + 1]
                    mask = np.random.random(dim) < 0.5
                    child1 = np.where(mask, parent1, parent2)
                    child2 = np.where(mask, parent2, parent1)
                    new_population[i], new_population[i + 1] = child1, child2
            
            # 变异
            for i in range(population_size):
                if np.random.random() < 0.1:  # 变异概率
                    for j in range(dim):
                        low, high = self.DRONE_BOUNDS[j]
                        new_population[i][j] += np.random.normal(0, (high - low) / 10)
                        new_population[i][j] = np.clip(new_population[i][j], low, high)
            
            population = np.array(new_population)
            pbar.update(1)
        
        pbar.close()
        logging.info(f"优化完成。")
        logging.info(f"找到的最优策略可实现最长总遮蔽时间: {best_fitness:.2f} 秒。")
        return self._format_and_save_output(best_individual, prefix='final_')

    def resume_from_checkpoint(self, population_size=50, total_generations=500, 
                              checkpoint_interval=50, checkpoint_path='Data/Q3/checkpoint.npy'):
        """从检查点文件恢复优化"""
        if population_size < 5:
            logging.warning("种群大小必须大于4，已自动设置为5。")
            population_size = 5

        logging.info(f"正在从检查点文件 {checkpoint_path} 读取模型...")
        if not os.path.exists(checkpoint_path):
            logging.error("检查点文件不存在，无法恢复。")
            return

        try:
            checkpoint_data = np.load(checkpoint_path, allow_pickle=True).item()
            start_generation = checkpoint_data['generation']
            initial_params = checkpoint_data['best_params']
            
            if len(initial_params) != len(self.DRONE_BOUNDS):
                raise ValueError(f"检查点中的 best_params 维度 {len(initial_params)} 与预期 {len(self.DRONE_BOUNDS)} 不匹配")
            for i, (val, (low, high)) in enumerate(zip(initial_params, self.DRONE_BOUNDS)):
                if not (low <= val <= high):
                    raise ValueError(f"检查点中的 best_params[{i}]={val} 超出边界 [{low}, {high}]")
            logging.info(f"检查点 best_params: {initial_params}")
        except Exception as e:
            logging.error(f"加载检查点文件失败。可能是文件已损坏。请尝试删除 {checkpoint_path} 后重新运行程序以开始新的优化。")
            logging.error(f"具体错误信息: {e}")
            return
        
        logging.info(f"已从第 {start_generation} 次迭代恢复。")
        
        dim = len(self.DRONE_BOUNDS)
        init_population = np.zeros((population_size, dim))
        for i in range(dim):
            low, high = self.DRONE_BOUNDS[i]
            init_population[:, i] = np.random.uniform(low, high, population_size)
        init_population[0] = initial_params
        
        if init_population.shape != (population_size, dim):
            raise ValueError(f"初始种群形状 {init_population.shape} 不符合预期 ({population_size}, {dim})")
        
        logging.info(f"init_population shape: {init_population.shape}")
        logging.info(f"init_population sample:\n{init_population[:5]}")
        
        self.genetic_algorithm(
            population_size=population_size,
            max_generations=total_generations - start_generation,
            checkpoint_interval=checkpoint_interval,
            init_population=init_population
        )

# --- 使用示例 ---
if __name__ == '__main__':
    optimizer = DroneObscurationOptimizer()
    
    # 定义优化参数
    POP_SIZE = 50
    MAX_GENERATIONS = 500
    CHECKPOINT_INTERVAL = 50
    CHECKPOINT_PATH = 'Data/Q3/checkpoint.npy'
    
    # 自动检测检查点并决定运行模式
    if os.path.exists(CHECKPOINT_PATH):
        logging.info("检测到检查点文件，将从上次中断处继续优化...")
        optimizer.resume_from_checkpoint(
            population_size=POP_SIZE,
            total_generations=MAX_GENERATIONS,
            checkpoint_interval=CHECKPOINT_INTERVAL,
            checkpoint_path=CHECKPOINT_PATH
        )
    else:
        logging.info("未检测到检查点文件，将开始新的优化...")
        final_results_df = optimizer.genetic_algorithm(
            population_size=POP_SIZE,
            max_generations=MAX_GENERATIONS,
            checkpoint_interval=CHECKPOINT_INTERVAL
        )
        logging.info("\n--- 最终最优策略详情 ---")
        print(final_results_df.to_string())