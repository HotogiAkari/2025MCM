import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random
from tqdm import tqdm
import os
import math

# 常数定义
g = 9.8  # 重力加速度 (m/s^2)
v_drone_min, v_drone_max = 70, 140  # 无人机速度范围 (m/s)
smoke_radius = 10  # 烟幕有效半径 (m)
smoke_duration = 20  # 烟幕持续时间 (s)
smoke_sink_speed = 3  # 烟幕下沉速度 (m/s)
target_pos = np.array([0, 200, 0])  # 真目标位置
drone_pos0 = np.array([17800, 0, 1800])  # FY1初始位置

# 计算烟幕云团位置
def smoke_position(release_pos, release_time, burst_time, t, v, direction):
    dt = t - burst_time
    return (release_pos + (burst_time - release_time) * v * direction + 
            np.array([0, 0, -0.5 * g * (burst_time - release_time)**2]) + 
            np.array([0, 0, -smoke_sink_speed * dt]))

# 计算有效干扰时长
def calc_coverage_time(release_pos, release_time, burst_time, v, direction):
    total_time = 0
    dt = 0.1
    for t in np.arange(0, smoke_duration, dt):
        smoke_pos = smoke_position(release_pos, release_time, burst_time, t, v, direction)
        if np.linalg.norm(smoke_pos - target_pos) <= smoke_radius:
            total_time += dt
    return min(total_time, smoke_duration)  # 限制在20s内

# 适应度函数
def evaluate(individual, v, direction):
    v, dx, dy, t1, t2, t3, dt1, dt2, dt3 = individual
    norm = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / norm, dy / norm
    direction = np.array([dx, dy, 0])
    
    total_coverage = 0
    for ti, dti in [(t1, dt1), (t2, dt2), (t3, dt3)]:
        drone_pos = drone_pos0 + ti * v * direction
        burst_time = ti + dti
        total_coverage += calc_coverage_time(drone_pos, ti, burst_time, v, direction)
    
    # 惩罚违反时间约束
    penalty = 0
    if t2 < t1 + 1 or t3 < t2 + 1:
        penalty = -1000 * (max(0, t1 + 1 - t2) + max(0, t2 + 1 - t3))
    return total_coverage + penalty,

# 主函数
def optimize_smoke_strategy():
    # 设置遗传算法
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_v", random.uniform, v_drone_min, v_drone_max)
    toolbox.register("attr_dir", random.uniform, -1, 1)
    toolbox.register("attr_t", random.uniform, 0, 50)
    toolbox.register("attr_dt", random.uniform, 0, 5)
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_v, toolbox.attr_dir, toolbox.attr_dir,
                      toolbox.attr_t, toolbox.attr_t, toolbox.attr_t,
                      toolbox.attr_dt, toolbox.attr_dt, toolbox.attr_dt), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, v=None, direction=None)  # 占位符，实际在循环中传入
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # 运行遗传算法并添加进度条
    pop = toolbox.population(n=100)
    with tqdm(total=100, desc="Optimizing Strategy") as pbar:
        for gen in range(100):
            pop = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.3)
            fits = list(map(lambda x: evaluate(x, x[0], np.array([x[1], x[2], 0]) / np.sqrt(x[1]**2 + x[2]**2)), pop))
            for fit, ind in zip(fits, pop):
                ind.fitness.values = fit
            pop = toolbox.select(pop, k=len(pop))
            pbar.update(1)

    # 获取最优解
    best_ind = tools.selBest(pop, 1)[0]
    v, dx, dy, t1, t2, t3, dt1, dt2, dt3 = best_ind
    norm = np.sqrt(dx**2 + dy**2)
    dx, dy = dx / norm, dy / norm
    # 计算运动方向（0~360°，逆时针为正）
    direction_angle = math.degrees(math.atan2(dy, dx))
    if direction_angle < 0:
        direction_angle += 360
    direction = np.array([dx, dy, 0])

    # 计算投放点和起爆点及有效干扰时长
    data = []
    for i, (ti, dti) in enumerate([(t1, dt1), (t2, dt2), (t3, dt3)], 1):
        release_pos = drone_pos0 + ti * v * direction
        burst_time = ti + dti
        burst_pos = (release_pos + dti * v * direction + 
                     np.array([0, 0, -0.5 * g * dti**2]))
        coverage_time = calc_coverage_time(release_pos, ti, burst_time, v, direction)
        data.append([direction_angle, v, i, release_pos[0], release_pos[1], release_pos[2],
                     burst_pos[0], burst_pos[1], burst_pos[2], coverage_time])

    # 确保Data目录存在
    os.makedirs('../Data', exist_ok=True)
    # 保存到Excel
    df = pd.DataFrame(data, columns=['无人机运动方向 (度)', '无人机运动速度 (m/s)', '烟幕干扰弹编号',
                                    '烟幕干扰弹投放点x坐标 (m)', '烟幕干扰弹投放点y坐标 (m)', 
                                    '烟幕干扰弹投放点z坐标 (m)', '烟幕干扰弹起爆点x坐标 (m)', 
                                    '烟幕干扰弹起爆点y坐标 (m)', '烟幕干扰弹起爆点z坐标 (m)', 
                                    '有效干扰时长 (s)'])
    df.to_excel('../Data/result1.xlsx', index=False)

    print("优化完成，结果已保存到 ../Data/result1.xlsx")

if __name__ == "__main__":
    optimize_smoke_strategy()