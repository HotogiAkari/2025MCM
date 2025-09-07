# -*- coding: utf-8 -*-
"""
问题3：使用FY1投放3枚烟幕弹干扰M1
优化方法：差分进化搜索最佳航向、速度、投放时机和爆炸延迟
功能：
1. 保存产生最优解的随机种子，以便复现。
2. 采用“先粗略后精细”的三阶段优化策略。
3. 调整Excel输出格式，使其更清晰。
4. 改进了烟幕干扰判定条件：判定真目标完全处于导弹与烟雾组成的锥体内。
5. 增加实时进度条，方便监控优化过程。
6. 新增功能：优化烟幕弹投掷后爆炸延迟时间。
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import differential_evolution

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------
# 设备选择：xp = cupy or numpy
# ---------------------------
try:
    import cupy as cp
    try:
        devcount = cp.cuda.runtime.getDeviceCount()
    except Exception:
        devcount = 0
    if devcount > 0:
        xp = cp
        USE_GPU = True
        print(f"使用 CuPy (GPU)，设备数量: {devcount}")
    else:
        xp = np
        USE_GPU = False
        print("未检测到可用 GPU，使用 NumPy")
except Exception:
    xp = np
    USE_GPU = False
    print("无法导入 CuPy，使用 NumPy")

# ---------------------------
# 仿真 / 物理 参数（可调整）
# ---------------------------
# 导弹 (M1)：以固定的速度和轨迹飞行
MISSILE_SPEED = 300.0   # m/s
MISSILE_INIT = np.array([20000.0, 0.0, 2000.0]) # 初始位置
# 目标 (M1)：圆柱形，位于 (0,0,0)
TARGET_CENTER = np.array([0.0, 0.0, 0.0])
TARGET_RADIUS = 7.0
TARGET_HEIGHT = 10.0

# 无人机 (FY1)：从固定初始位置出发，航向和速度确定后不再改变
UAV_INIT = np.array([17800.0, 0.0, 1800.0])

# 烟幕弹：共3枚，每个具有固定物理特性
SMOKE_RADIUS = 10.0     # 烟雾半径
SMOKE_LIFE = 20.0       # 烟雾持续时间
SMOKE_SINK = 3.0        # 烟雾下沉速度
NUM_BOMBS = 3

# 物理常数与仿真参数
GRAVITY = 9.8           # 重力加速度，m/s^2
DT = 0.1                # 仿真时间步长

# 烟幕弹自由落体到地面所需时间
T_FALL_MAX = np.sqrt(2 * UAV_INIT[2] / GRAVITY)

# 最大模拟时间（基于导弹到目标的飞行时间）
T_MAX = np.linalg.norm(TARGET_CENTER - MISSILE_INIT) / MISSILE_SPEED
T_MAX = float(np.round(T_MAX, 1) + 1.0)

# 转换到 xp 变量（统一）
TARGET_CENTER_XP = xp.array(TARGET_CENTER, dtype=xp.float64)
TARGET_RADIUS_XP = xp.array(TARGET_RADIUS, dtype=xp.float64)
TARGET_HEIGHT_XP = xp.array(TARGET_HEIGHT, dtype=xp.float64)
MISSILE_INIT_XP = xp.array(MISSILE_INIT, dtype=xp.float64)
UAV_INIT_XP = xp.array(UAV_INIT, dtype=xp.float64)
SMOKE_RADIUS_XP = xp.array(SMOKE_RADIUS, dtype=xp.float64)

# 时间网格（xp）
TIMES_XP = xp.arange(0.0, T_MAX + 1e-9, DT, dtype=xp.float64)
NUM_TIMES = int(TIMES_XP.shape[0])

# 预计算导弹轨迹（xp）
def _missile_pos_vec(times_xp):
    dir_vec = (TARGET_CENTER_XP - MISSILE_INIT_XP)
    dir_vec = dir_vec / xp.linalg.norm(dir_vec)
    return MISSILE_INIT_XP[None, :] + xp.outer(times_xp, dir_vec * MISSILE_SPEED)

MISSILE_TRAJ_XP = _missile_pos_vec(TIMES_XP)  # shape (T,3)

# ---------------------------
# 基础工具（都用 xp）
# ---------------------------
def normalize_xy(dx, dy):
    """
    输入 dx, dy（Python 数值或 numpy 标量），在 xp 上返回单位向量 (3,)
    完全使用 xp API，避免 numpy/cupy 混合转换问题。
    """
    arr = xp.asarray([dx, dy], dtype=xp.float64)  # xp array
    n = xp.linalg.norm(arr) + 1e-12
    first2 = arr / n
    # 拼接第三个分量 0.0（xp array）
    return xp.concatenate([first2, xp.array([0.0], dtype=xp.float64)])  # shape (3,)

# ---------------------------
# 向量化遮蔽判断（单炸弹）
# ---------------------------
def vectorized_block_mask_for_bomb(t_b, pos_b, t_d):
    """
    t_b: 投放时间，float (Python)
    pos_b: 投放位置，3-element array-like (could be xp array or numpy)
    t_d: 爆炸延迟时间，float (Python)
    返回：xp boolean array shape (T,) 表示哪些时间点“完全遮挡目标”
    算法：中心剪枝 + 圆柱8点检查（向量化）
    """
    pos = xp.asarray(pos_b, dtype=xp.float64)
    # 烟幕有效时间段从 t_b + t_d 开始
    effective_start_time = float(t_b) + float(t_d)
    active_mask = (TIMES_XP >= effective_start_time) & (TIMES_XP <= effective_start_time + SMOKE_LIFE)
    
    if not xp.any(active_mask):
        return xp.zeros_like(TIMES_XP, dtype=xp.bool_)

    times_active = TIMES_XP[active_mask]        # (M,)
    m_pos = MISSILE_TRAJ_XP[active_mask]        # (M,3)

    # 烟幕中心随时间的 z 变化
    smoke_z = pos[2] - SMOKE_SINK * (times_active - effective_start_time)
    smoke_centers = xp.stack([
        xp.full_like(times_active, pos[0]),
        xp.full_like(times_active, pos[1]),
        smoke_z
    ], axis=1)  # (M,3)

    # 快速剪枝：导弹->目标中心线段到 smoke_center 的最短距离
    p1 = m_pos
    p2 = xp.broadcast_to(TARGET_CENTER_XP, p1.shape)
    v = p2 - p1
    w = smoke_centers - p1
    vv = xp.sum(v * v, axis=1) + 1e-9
    tproj = xp.clip(xp.sum(w * v, axis=1) / vv, 0.0, 1.0)
    closest = p1 + (tproj[:, None] * v)
    d_center = xp.linalg.norm(closest - smoke_centers, axis=1)
    center_blocked = d_center <= SMOKE_RADIUS_XP

    if not xp.any(center_blocked):
        return xp.zeros_like(TIMES_XP, dtype=xp.bool_)

    # 对 center_blocked 时刻检查圆柱的 8 个关键点
    top_z = TARGET_CENTER_XP[2] + TARGET_HEIGHT_XP / 2.0
    bottom_z = TARGET_CENTER_XP[2] - TARGET_HEIGHT_XP / 2.0
    circ_pts = xp.array([
        [TARGET_CENTER_XP[0] + TARGET_RADIUS_XP, TARGET_CENTER_XP[1], top_z],
        [TARGET_CENTER_XP[0] - TARGET_RADIUS_XP, TARGET_CENTER_XP[1], top_z],
        [TARGET_CENTER_XP[0], TARGET_CENTER_XP[1] + TARGET_RADIUS_XP, top_z],
        [TARGET_CENTER_XP[0], TARGET_CENTER_XP[1] - TARGET_RADIUS_XP, top_z],
        [TARGET_CENTER_XP[0] + TARGET_RADIUS_XP, TARGET_CENTER_XP[1], bottom_z],
        [TARGET_CENTER_XP[0] - TARGET_RADIUS_XP, TARGET_CENTER_XP[1], bottom_z],
        [TARGET_CENTER_XP[0], TARGET_CENTER_XP[1] + TARGET_RADIUS_XP, bottom_z],
        [TARGET_CENTER_XP[0], TARGET_CENTER_XP[1] - TARGET_RADIUS_XP, bottom_z]
    ], dtype=xp.float64)  # (8,3)

    idxs = xp.where(center_blocked)[0]
    if idxs.size == 0:
        return xp.zeros_like(TIMES_XP, dtype=xp.bool_)

    p1_sel = p1[idxs]               # (M_sel,3)
    smoke_sel = smoke_centers[idxs] # (M_sel,3)
    p2_sel = xp.broadcast_to(TARGET_CENTER_XP, p1_sel.shape)
    v_sel = p2_sel - p1_sel
    denom = xp.linalg.norm(v_sel, axis=1) + 1e-9

    rel_pts = circ_pts - TARGET_CENTER_XP      # (8,3)
    v_exp = v_sel[:, None, :]                  # (M_sel,1,3)
    rel_exp = xp.broadcast_to(rel_pts[None, :, :], (v_sel.shape[0], rel_pts.shape[0], 3))  # (M_sel,8,3)
    cross_prod = xp.cross(v_exp, rel_exp, axis=2)  # (M_sel,8,3)
    num = xp.linalg.norm(cross_prod, axis=2)       # (M_sel,8)
    dists = num / (denom[:, None] + 1e-9)          # (M_sel,8)

    fully_blocked_sel = xp.all(dists <= SMOKE_RADIUS_XP, axis=1)  # (M_sel,)

    res_mask = xp.zeros_like(TIMES_XP, dtype=xp.bool_)
    active_indices = xp.where(active_mask)[0]
    set_indices = active_indices[idxs]  # indices in TIMES_XP
    res_mask[set_indices] = fully_blocked_sel
    return res_mask

# ---------------------------
# 聚合遮蔽（所有炸弹）
# ---------------------------
def simulate_coverage_from_bombs(bombs):
    """
    bombs: list of (t_b, pos_b, t_d)
    返回 (total_coverage_xp_scalar, list_of_per_bomb_xp_scalar)
    """
    if not bombs:
        return xp.array(0.0, dtype=xp.float64), [xp.array(0.0, dtype=xp.float64)] * 0

    blocked_any = xp.zeros_like(TIMES_XP, dtype=xp.bool_)
    per_masks = []
    for (t_b, pos_b, t_d) in bombs:
        mask = vectorized_block_mask_for_bomb(t_b, pos_b, t_d)
        per_masks.append(mask)
        blocked_any = blocked_any | mask

    total_coverage = xp.sum(blocked_any) * DT
    per_cov = [xp.sum(m) * DT for m in per_masks]
    return total_coverage, per_cov

def simulate_single_coverage_scalar(t_b, pos_b, t_d):
    mask = vectorized_block_mask_for_bomb(t_b, pos_b, t_d)
    return xp.sum(mask) * DT

# ---------------------------
# 参数映射（params -> bombs）
# ---------------------------
# 优化参数格式：
# [dx, dy, v, t1, d1, t2, d2, t3, d3]
# dx, dy：无人机二维航向向量分量
# v：无人机速度
# ti：第 i 枚烟幕弹的投放时机
# di：第 i 枚烟幕弹的爆炸延迟时间
def params_to_bombs(params, num_bombs=NUM_BOMBS):
    """
    params: numpy array-like [dx, dy, v, t1, d1, t2, d2, t3, d3]
    返回：list of (t_b (float), pos_b xp array(3,), t_d (float))
    内部把必要量转换为 xp，保证一致性
    """
    dx = float(params[0]); dy = float(params[1]); v = float(params[2])
    
    bombs = []
    dir_vec = normalize_xy(dx, dy)  # xp vector (3,)
    
    for i in range(num_bombs):
        t_drop = float(params[3 + 2 * i])
        t_delay = float(params[4 + 2 * i])
        pos = UAV_INIT_XP + dir_vec * (v * t_drop)
        bombs.append((t_drop, pos, t_delay))
        
    return bombs

def get_coverage_from_params(params, num_bombs=NUM_BOMBS):
    """
    params: numpy array-like
    返回 (total_coverage_float, per_drop_dict)
    """
    bombs = params_to_bombs(params, num_bombs)
    per_drop = {}
    for i, (t_b, pos_b, t_d) in enumerate(bombs):
        cov = simulate_single_coverage_scalar(t_b, pos_b, t_d)
        per_drop[f'Bomb-{i+1}'] = float(xp.asnumpy(cov)) if USE_GPU else float(cov)
        
    total_cov_xp, per_cov_list = simulate_coverage_from_bombs(bombs)
    total_cov = float(xp.asnumpy(total_cov_xp)) if USE_GPU else float(total_cov_xp)
    return total_cov, per_drop

# ---------------------------
# 目标函数（无副作用）
# ---------------------------
# 优化目标：最大化烟幕弹产生的总有效遮蔽时间
# 约束条件：相邻两次投掷的时间间隔必须大于1秒
def global_evaluate(params):
    """
    粗略目标：-sum(each bomb independent coverage)
    params: numpy array-like [dx,dy,v,t1,d1,...]
    """
    # 按照投放时间 t_drop 排序
    t_drops = np.array([params[3 + 2 * i] for i in range(NUM_BOMBS)])
    sorted_idx = np.argsort(t_drops)
    
    sorted_params = np.array(params, dtype=float)
    for i, original_idx in enumerate(sorted_idx):
        sorted_params[3 + 2 * i] = params[3 + 2 * original_idx]
        sorted_params[4 + 2 * i] = params[4 + 2 * original_idx]
        
    # 投放时间间隔太近则惩罚
    if np.any(np.diff(sorted_params[3::2]) < 1.0):
        return 1e6

    bombs = params_to_bombs(sorted_params, NUM_BOMBS)
    total_individual = 0.0
    for (t_b, pos_b, t_d) in bombs:
        cov = simulate_single_coverage_scalar(t_b, pos_b, t_d)
        total_individual += float(xp.asnumpy(cov)) if USE_GPU else float(cov)
    return float(-total_individual)

def final_evaluate(params):
    """
    精细目标：-total coverage considering overlap
    params: numpy array-like
    """
    # 按照投放时间 t_drop 排序
    t_drops = np.array([params[3 + 2 * i] for i in range(NUM_BOMBS)])
    sorted_idx = np.argsort(t_drops)
    
    sorted_params = np.array(params, dtype=float)
    for i, original_idx in enumerate(sorted_idx):
        sorted_params[3 + 2 * i] = params[3 + 2 * original_idx]
        sorted_params[4 + 2 * i] = params[4 + 2 * original_idx]
        
    # 投放时间间隔太近则惩罚
    if np.any(np.diff(sorted_params[3::2]) < 1.0):
        return 1e6

    bombs = params_to_bombs(sorted_params, NUM_BOMBS)
    total_cov_xp, _ = simulate_coverage_from_bombs(bombs)
    total_cov = float(xp.asnumpy(total_cov_xp)) if USE_GPU else float(total_cov_xp)
    return float(-total_cov)

# ---------------------------
# 随机预热（warm start）
# ---------------------------
def random_warm_start(n_samples=500, num_bombs=NUM_BOMBS):
    """
    随机采样求 warm-start（返回 numpy params 与对应覆盖）
    params format: [dx,dy,v,t1,d1,...]
    """
    best_score = -1e9
    best_params = None
    # sample in numpy (optimizer wants numpy)
    dxs = np.random.uniform(-1.0, 1.0, size=n_samples)
    dys = np.random.uniform(-1.0, 1.0, size=n_samples)
    vs = np.random.uniform(70.0, 140.0, size=n_samples)
    
    t_drops = np.random.uniform(0.0, T_MAX, size=(n_samples, num_bombs))
    t_delays = np.random.uniform(0.0, T_FALL_MAX, size=(n_samples, num_bombs))
    
    for k in range(n_samples):
        params_k = np.zeros(3 + 2 * num_bombs)
        params_k[0] = dxs[k]
        params_k[1] = dys[k]
        params_k[2] = vs[k]
        params_k[3::2] = t_drops[k]
        params_k[4::2] = t_delays[k]
        
        # evaluate using final_evaluate (which handles numpy input)
        cov = -final_evaluate(params_k)
        if cov > best_score:
            best_score = cov
            best_params = params_k.copy()
            
    return best_params, best_score

# ---------------------------
# 历史保存、Excel 输出
# ---------------------------
def _load_optimization_data(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    except Exception:
        return {'best_seed': None, 'best_coverage': -1.0, 'best_params': None, 'last_tried_seed': -1}

def _save_optimization_data(data, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def generate_excel_rows(best_params, per_drop_coverage):
    if best_params is None:
        return []
    
    dx = float(best_params[0]); dy = float(best_params[1]); v = float(best_params[2])
    
    # 提取投放时间和延迟时间
    t_drops = np.array([best_params[3 + 2 * i] for i in range(NUM_BOMBS)], dtype=float)
    t_delays = np.array([best_params[4 + 2 * i] for i in range(NUM_BOMBS)], dtype=float)
    
    # 排序以美化输出
    sorted_with_idx = sorted(list(enumerate(t_drops)), key=lambda x: x[1])
    
    direction_vec = np.array([dx, dy, 0.0])
    nrm = np.linalg.norm(direction_vec) + 1e-12
    direction_vec = direction_vec / nrm
    
    rows = []
    for i, (orig_idx, t_drop) in enumerate(sorted_with_idx):
        pos = (np.array(xp.asnumpy(UAV_INIT_XP)) + direction_vec * (v * t_drop))
        bomb_name = f'Bomb-{orig_idx+1}'
        cover_time = per_drop_coverage.get(bomb_name, 0.0)
        
        rows.append({
            '无人机运动方向_dx': dx,
            '无人机运动方向_dy': dy,
            '无人机运动速度 (m/s)': v,
            '烟幕干扰弹名称': i + 1,
            '烟幕干扰弹投放点的x坐标 (m)': float(pos[0]),
            '烟幕干扰弹投放点的y坐标 (m)': float(pos[1]),
            '烟幕干扰弹投放点的z坐标 (m)': float(pos[2]),
            '投放时机 (s)': float(t_drop),
            '爆炸延迟时间 (s)': float(t_delays[orig_idx]),
            '有效干扰时长 (s)': float(cover_time)
        })
        
    return rows

# ---------------------------
# DE 回调（进度条）
# ---------------------------
pbar = None
def de_callback(xk, convergence=None):
    global pbar
    if pbar is not None:
        try:
            pbar.update(1)
        except Exception:
            pass

# ---------------------------
# 主流程：三阶段 DE + 随机预热 + 局部优化
# ---------------------------
def run_optimization(num_runs=3,
                     max_iter_stage1=200,
                     max_iter_stage2=80,
                     max_iter_stage3=30,
                     seed_file_path="Data/Q3/optimization_seed.pkl",
                     use_random_warmup=True,
                     warmup_samples=400):
    global pbar
    history = _load_optimization_data(seed_file_path)

    for run_num in range(1, num_runs + 1):
        print("\n" + "="*60)
        print(f"开始第 {run_num}/{num_runs} 次优化运行")
        print("="*60)
        last_seed = history.get('last_tried_seed', -1)
        current_seed = last_seed + 1
        print(f"随机种子: {current_seed}")
        np.random.seed(current_seed)

        # 随机预热
        x0_guess = None
        if use_random_warmup:
            print("随机预热（warm start）中...")
            x0_guess, warm_score = random_warm_start(n_samples=warmup_samples, num_bombs=NUM_BOMBS)
            print(f"warm start 覆盖: {warm_score:.3f}")

        # Stage 1: 粗略优化
        print("\n=== 阶段1：粗略优化（global evaluate） ===")
        # 优化参数：[dx, dy, v, t1, d1, t2, d2, t3, d3]
        bounds = [
            (-1.0, 1.0),      # dx
            (-1.0, 1.0),      # dy
            (70.0, 140.0)     # v
        ] + [(0.0, T_MAX), (0.0, T_FALL_MAX)] * NUM_BOMBS

        pbar = tqdm(total=max_iter_stage1, desc="粗略优化")
        rough_result = differential_evolution(
            global_evaluate,
            bounds,
            maxiter=max_iter_stage1,
            popsize=40,
            polish=True,
            disp=False,
            seed=current_seed,
            callback=de_callback
        )
        pbar.close()
        rough_best_params = rough_result.x.copy()
        total_cov_rough, per_drop_rough = get_coverage_from_params(rough_best_params, NUM_BOMBS)
        print("阶段1 完成：")
        print(f"  dx,dy = {rough_best_params[0]:.4f}, {rough_best_params[1]:.4f}, v = {rough_best_params[2]:.2f}")
        print(f"  各炸弹独立贡献和(approx): {sum(per_drop_rough.values()):.3f} s")
        print(f"  实际总遮蔽(粗略): {total_cov_rough:.3f} s")

        # Stage 2: 精细优化
        print("\n=== 阶段2：精细优化（final evaluate） ===")
        pbar = tqdm(total=max_iter_stage2, desc="精细优化")
        x0_to_use = rough_best_params.copy()
        if x0_guess is not None:
            # 使用随机预热结果作为初始猜想
            x0_to_use[0:3] = x0_guess[0:3]
            for i in range(NUM_BOMBS):
                x0_to_use[3 + 2 * i] = x0_guess[3 + 2 * i]
                x0_to_use[4 + 2 * i] = x0_guess[4 + 2 * i]
                
        fine_result = differential_evolution(
            final_evaluate,
            bounds,
            maxiter=max_iter_stage2,
            popsize=20,
            x0=x0_to_use,
            polish=True,
            disp=False,
            seed=current_seed,
            callback=de_callback
        )
        pbar.close()
        final_best_params = fine_result.x.copy()
        final_total_cov, final_per_drop = get_coverage_from_params(final_best_params, NUM_BOMBS)
        print("阶段2 完成：")
        print(f"  dx,dy = {final_best_params[0]:.4f}, {final_best_params[1]:.4f}, v = {final_best_params[2]:.2f}")
        print(f"  最终总遮蔽: {final_total_cov:.3f} s")
        print(f"  每炸弹贡献: { {k: f'{v:.3f}' for k,v in final_per_drop.items()} }")

        # Stage 3: 针对性局部优化
        # 找出贡献度低的炸弹，只优化其投放时间和延迟
        ineffective = [i for i, (k, v) in enumerate(final_per_drop.items()) if v < 0.1]
        if ineffective:
            print("\n=== 阶段3：针对性局部优化 ===")
            for idx in ineffective:
                bomb_name = f'Bomb-{idx+1}'
                initial_t = final_best_params[3 + 2 * idx]
                initial_d = final_best_params[4 + 2 * idx]
                print(f"  局部优化 {bomb_name} 初始时间 {initial_t:.2f}s, 延迟 {initial_d:.2f}s")
                
                def targeted_eval_wrapper(td_arr, params_fixed, bomb_idx):
                    params_tmp = np.array(params_fixed, dtype=float)
                    params_tmp[3 + 2 * bomb_idx] = float(td_arr[0])
                    params_tmp[4 + 2 * bomb_idx] = float(td_arr[1])
                    return final_evaluate(params_tmp)
                    
                local_bounds = [
                    (max(0.0, initial_t - 5.0), min(T_MAX, initial_t + 5.0)),
                    (max(0.0, initial_d - 5.0), min(T_FALL_MAX, initial_d + 5.0))
                ]
                
                pbar = tqdm(total=max_iter_stage3, desc=f"局部优化 {bomb_name}")
                local_res = differential_evolution(
                    lambda x: targeted_eval_wrapper(x, final_best_params, idx),
                    local_bounds,
                    maxiter=max_iter_stage3,
                    popsize=10,
                    polish=True,
                    disp=False,
                    seed=current_seed,
                    callback=de_callback
                )
                pbar.close()
                final_best_params[3 + 2 * idx] = local_res.x[0]
                final_best_params[4 + 2 * idx] = local_res.x[1]
            final_total_cov, final_per_drop = get_coverage_from_params(final_best_params, NUM_BOMBS)
            print("局部优化后：")
            print(f"  总遮蔽: {final_total_cov:.3f} s")
            print(f"  每炸弹贡献: { {k: f'{v:.3f}' for k,v in final_per_drop.items()} }")

        # 更新历史
        hist_best = history.get('best_coverage', -1.0)
        if final_total_cov > hist_best:
            print("本次为历史新高，更新记录。")
            history['best_coverage'] = final_total_cov
            history['best_seed'] = current_seed
            history['best_params'] = final_best_params.copy()
        else:
            print(f"本次({final_total_cov:.3f}s)未超过历史最佳({hist_best:.3f}s)。")
        history['last_tried_seed'] = current_seed
        _save_optimization_data(history, seed_file_path)
        print("本次运行结束并保存历史。")

    return history.get('best_params', None), history.get('best_coverage', -1.0)

# ---------------------------
# 绘图（半透明球体）
# ---------------------------
def plot_results(best_params):
    if best_params is None:
        print("没有最优参数，无法绘图。")
        return
    
    # 解析最优参数
    bp = np.array(best_params, dtype=float)
    dx, dy, v = bp[0], bp[1], bp[2]
    t_drops = np.array([bp[3 + 2 * i] for i in range(NUM_BOMBS)], dtype=float)
    t_delays = np.array([bp[4 + 2 * i] for i in range(NUM_BOMBS)], dtype=float)

    # 计算无人机运动方向向量
    dir_vec = np.array([dx, dy, 0.0])
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-12)
    
    # 投放点位置
    bomb_positions = np.array([np.array(xp.asnumpy(UAV_INIT_XP)) + dir_vec * (v * t) for t in t_drops])

    # 导弹轨迹 (cpu)
    times = np.arange(0.0, T_MAX + 1e-9, DT)
    dir_m = (np.array(TARGET_CENTER) - np.array(MISSILE_INIT))
    dir_m = dir_m / np.linalg.norm(dir_m)
    missile_traj = np.array(MISSILE_INIT) + np.outer(times, dir_m * MISSILE_SPEED)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(missile_traj[:,0], missile_traj[:,1], missile_traj[:,2], 'r-', label='导弹轨迹')

    # 无人机路径
    drone_traj = np.array([np.array(xp.asnumpy(UAV_INIT_XP)) + dir_vec * (v * t) for t in times])
    ax.plot(drone_traj[:,0], drone_traj[:,1], drone_traj[:,2], 'b--', label='无人机轨迹')

    # 半透明球体表示烟幕爆炸点
    u = np.linspace(0, 2*np.pi, 30)
    v_ang = np.linspace(0, np.pi, 15)
    uu, vv_ang = np.meshgrid(u, v_ang)
    for i, pos in enumerate(bomb_positions):
        x = pos[0] + SMOKE_RADIUS * np.cos(uu) * np.sin(vv_ang)
        y = pos[1] + SMOKE_RADIUS * np.sin(uu) * np.sin(vv_ang)
        z = pos[2] + SMOKE_RADIUS * np.cos(vv_ang)
        ax.plot_surface(x, y, z, color='green', alpha=0.12, linewidth=0)

    # 投放点标记
    ax.scatter(bomb_positions[:,0], bomb_positions[:,1], bomb_positions[:,2], c='g', s=60, label='烟幕弹投放点')
    ax.scatter(np.array(UAV_INIT)[0], np.array(UAV_INIT)[1], np.array(UAV_INIT)[2], marker='^', s=80, label='无人机起始点')
    ax.scatter(np.array(TARGET_CENTER)[0], np.array(TARGET_CENTER)[1], np.array(TARGET_CENTER)[2], marker='x', s=80, label='目标')

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend()
    plt.title("导弹轨迹 / 无人机轨迹 / 烟幕示意")
    plt.show()

# ---------------------------
# main
# ---------------------------
if __name__ == "__main__":
    SEED_FILE = "Data/Q3/optimization_seed.pkl"
    EXCEL_PATH = "Data/Q3/result.xlsx"
    NUM_RUNS = 2
    MAX_ITER_STAGE1 = 200   # 调试时可设小一点
    MAX_ITER_STAGE2 = 80
    MAX_ITER_STAGE3 = 30

    best_params, best_cov = run_optimization(
        num_runs=NUM_RUNS,
        max_iter_stage1=MAX_ITER_STAGE1,
        max_iter_stage2=MAX_ITER_STAGE2,
        max_iter_stage3=MAX_ITER_STAGE3,
        seed_file_path=SEED_FILE,
        use_random_warmup=True,
        warmup_samples=300
    )

    print("\n==== 所有运行完成 ====")
    print("历史最佳覆盖 (s):", best_cov)
    if best_params is not None:
        total_cov, per_drop = get_coverage_from_params(best_params, NUM_BOMBS)
        print("最终最佳参数 (dx,dy,v,t1,d1,...):", np.round(best_params, 4))
        print("每个烟幕贡献:", {k: f"{v:.3f}" for k,v in per_drop.items()})

        # 保存到 Excel
        rows = generate_excel_rows(best_params, per_drop)
        if rows:
            os.makedirs(os.path.dirname(EXCEL_PATH), exist_ok=True)
            df = pd.DataFrame(rows)
            df.to_excel(EXCEL_PATH, index=False)
            print("Excel 已保存到：", EXCEL_PATH)
        else:
            print("没有生成 Excel 行。")

        # 绘图
        try:
            plot_results(best_params)
        except Exception as e:
            print("绘图失败：", e)
    else:
        print("未找到最佳参数，结束。")
