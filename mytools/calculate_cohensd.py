#!/usr/bin/env python3
"""
计算Cohen's d统计量，用于比较不同实验配置的效果

使用方法：
conda activate b2d_zoo
python calculate_cohensd.py [--model-type vad|uniad]

- VAD：从 open_loop_eval/<model>_rep_val.log 读取 plan_L2_* / plan_obj_box_col_*。
- UniAD：从 open_loop_eval/<model>_rep_val.json 用 baseline/convert_uniad_to_vad_metrics.py
  转成 VAD 格式的 summary 再解析（与 VAD 指标一致）。

实验结构：
- 两个方法：Arachne_v2, semSegRep
- 三个fitness函数：DISC, CONT, CONT2
- 每个实验会有若干次重复（repetition），数量以实际落盘文件为准
- 权重选择（wXX）以实际实验文件夹名解析到的为准（VAD常见 w26/52/105；UniAD可能是 w7/13/14 等）

提取的指标：
- plan_L2_1s, plan_L2_2s, plan_L2_3s (L2误差)
- plan_obj_box_col_1s, plan_obj_box_col_2s, plan_obj_box_col_3s (碰撞率)
"""

import os
import re
import json
import sys
import argparse
import subprocess
import tempfile

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: numpy and pandas are required. Please activate b2d_zoo conda environment:")
    print("  conda activate b2d_zoo")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from pathlib import Path
from collections import defaultdict

def extract_metrics_from_log(log_file):
    """
    从 VAD 格式的 log/summary 文件中提取指标。
    支持: plan_L2_1s:value 行格式，以及 convert_uniad_to_vad_metrics.py 输出的
    summary 格式（如 "  plan_L2_1s: 0.5 (n=100)"）。
    """
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        for horizon in [1, 2, 3]:
            # 支持 "plan_L2_1s:0.5" 以及 "plan_L2_1s: 0.5 (n=...)"（冒号后可有空格）
            pattern = rf'plan_L2_{horizon}s:\s*([\d.e+-]+)'
            matches = re.findall(pattern, content)
            if matches:
                # UniAD converter summary 里同一指标会出现两次（avg_all / avg_valid）。
                # 这里取“最后一次出现”，以便默认使用 avg_valid（有效帧）统计。
                metrics[f'plan_L2_{horizon}s'] = float(matches[-1].strip())
            else:
                metrics[f'plan_L2_{horizon}s'] = None
        for horizon in [1, 2, 3]:
            pattern = rf'plan_obj_box_col_{horizon}s:\s*([\d.e+-]+)'
            matches = re.findall(pattern, content)
            if matches:
                metrics[f'plan_obj_box_col_{horizon}s'] = float(matches[-1].strip())
            else:
                metrics[f'plan_obj_box_col_{horizon}s'] = None

        if any(metrics.get(f'plan_L2_{h}s') is not None for h in [1, 2, 3]):
            return metrics
        return None

    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None


def count_collision_frames_vad_rule(json_path):
    """
    从 converter 输出的 JSON（已按 VAD 规则设置 fut_valid_flag）中统计
    发生碰撞的帧数：有效帧且 plan_obj_box_col_3s > 0。
    与 baseline 588 帧的统计口径一致。
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        count = 0
        for item in data:
            if not item.get('fut_valid_flag'):
                continue
            v = item.get('plan_obj_box_col_3s')
            if isinstance(v, (int, float)) and v > 0:
                count += 1
        return count
    except Exception as e:
        return None


def parse_experiment_name(folder_name):
    """解析实验文件夹名称，提取配置信息"""
    # 格式示例: VAD_base_REP_VAL_Arachne_v2_DE_w26_p52_i50_es5_CONT_1
    # 或: VAD_base_REP_VAL_semSegRep_DE_w26_p52_i50_es5_CONT_1
    
    parts = folder_name.split('_')
    
    # 查找方法名
    method = None
    if 'Arachne_v2' in folder_name:
        method = 'Arachne_v2'
    elif 'semSegRep' in folder_name:
        method = 'semSegRep'
    
    # 查找权重数量
    weight_count = None
    for i, part in enumerate(parts):
        if part.startswith('w') and part[1:].isdigit():
            weight_count = int(part[1:])  # 提取数字部分
            break
    
    # 查找fitness函数类型（倒数第二个字段）
    fitness_type = None
    if len(parts) >= 2:
        fitness_type = parts[-2]
        if fitness_type not in ['DISC', 'CONT', 'CONT2']:
            fitness_type = None
    
    # 查找重复次数（最后一个字段）
    repetition = None
    if len(parts) >= 1:
        try:
            repetition = int(parts[-1])
        except ValueError:
            repetition = None
    
    return method, weight_count, fitness_type, repetition

def cohens_d(group1, group2, paired=False):
    """
    计算Cohen's d效应量
    
    Parameters:
    - group1, group2: 两组数据
    - paired: 是否为配对样本（paired samples）
    """
    if len(group1) == 0 or len(group2) == 0:
        return None
    
    if paired:
        # 配对样本的Cohen's d
        if len(group1) != len(group2):
            return None
        diffs = group1 - group2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
        if std_diff == 0:
            return None
        d = mean_diff / std_diff
    else:
        # 独立样本的Cohen's d
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1) if len(group1) > 1 else 0
        std2 = np.std(group2, ddof=1) if len(group2) > 1 else 0

        # 合并标准差
        n1, n2 = len(group1), len(group2)
        if n1 + n2 - 2 == 0:
            return None

        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return None

        d = (mean1 - mean2) / pooled_std

    return d

def interpret_cohens_d(d):
    """
    根据规则解释Cohen's d的效应量大小
    规则：
    - large_worse = d > 0.8        (group2更好，差异很大)
    - medium_worse = 0.2 <= d <= 0.8 (group2更好，中等差异)
    - small_worse = 0 < d < 0.2      (group2更好，小差异)
    - small_better = -0.2 < d < 0   (group1更好，小差异)
    - medium_better = -0.8 <= d <= -0.2 (group1更好，中等差异)
    - large_better = d < -0.8       (group1更好，差异很大)
    """
    if d is None:
        return "N/A"
    
    if d > 0.8:
        return "large_worse"
    elif d >= 0.2:
        return "medium_worse"
    elif d > 0:
        return "small_worse"
    elif d > -0.2:
        return "small_better"
    elif d >= -0.8:
        return "medium_better"
    else:  # d < -0.8
        return "large_better"

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算Cohen\'s d统计量')
    parser.add_argument('--time_horizon', type=str, choices=['1s', '2s', '3s'], default=None,
                        help='指定time horizon，只读取该目录下的数据（默认：读取所有）')
    parser.add_argument('--compare_horizons', action='store_true',
                        help='进行time horizon之间的比较（1s vs 2s vs 3s）')
    parser.add_argument('--model-type', type=str, choices=['vad', 'uniad'], default='vad',
                        help='模型类型：vad 或 uniad（默认 vad）')
    args = parser.parse_args()
    
    # 实验根目录（按模型类型）
    if args.model_type == 'vad':
        base_dirs = [
            '/data1/vad_base_Arachne_v2_DE_results',
            '/data1/vad_base_semSegRep_DE_results'
        ]
        folder_patterns = ['VAD_base_REP_VAL_*']
        def get_log_name(folder_name):
            return 'vad_base_rep_val.log'
    else:
        base_dirs = [
            '/data1/uniad_base_Arachne_v2_DE_results',
            '/data1/uniad_base_semSegRep_DE_results'
        ]
        folder_patterns = ['UniAD_base_REP_VAL_*', 'UniAD_tiny_REP_VAL_*']
        def get_log_name(folder_name):
            if 'UniAD_tiny' in folder_name or 'UniAD_tiny_REP' in folder_name:
                return 'uniad_tiny_rep_val.log'
            return 'uniad_base_rep_val.log'
    
    # 存储所有数据
    all_data = []
    # 存储time horizon数据：config -> repetition -> time_horizon -> metric -> value
    time_horizon_data = {}
    missing_log = []  # 缺少log文件的实验
    parse_failed = []  # 无法解析配置的实验
    extract_failed = []  # 无法提取指标的实验
    success_count = 0  # 成功提取的实验数
    
    print("=" * 80)
    print("Step 1: 提取所有实验的指标数据")
    if args.time_horizon:
        print(f"只读取 time_horizon={args.time_horizon} 的数据")
    print("=" * 80)
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            continue
        
        print(f"\n处理目录: {base_dir}")
        base_path = Path(base_dir)
        
        # 如果指定了time_horizon，只在该目录下查找
        if args.time_horizon:
            search_path = base_path / args.time_horizon
            if not search_path.exists():
                print(f"  Warning: {search_path} not found")
                continue
            exp_folders = []
            for pat in folder_patterns:
                for exp_folder in search_path.rglob(pat):
                    if exp_folder.is_dir():
                        exp_folders.append(exp_folder)
        else:
            exp_folders = []
            for pat in folder_patterns:
                for exp_folder in base_path.rglob(pat):
                    if exp_folder.is_dir():
                        exp_folders.append(exp_folder)
        exp_folders = sorted(set(exp_folders))
        print(f"  找到 {len(exp_folders)} 个实验文件夹")
        
        # UniAD 时用 convert_uniad_to_vad_metrics.py 把 JSON 转成 VAD 格式的 summary 再解析
        converter_script = Path(__file__).resolve().parent.parent / 'baseline' / 'convert_uniad_to_vad_metrics.py'

        for exp_folder in exp_folders:
            log_name = get_log_name(exp_folder.name)
            log_file = exp_folder / 'open_loop_eval' / log_name
            json_name = Path(log_name).stem + '.json'
            json_file = exp_folder / 'open_loop_eval' / json_name

            if args.model_type == 'vad':
                if not log_file.exists():
                    missing_log.append(str(exp_folder.relative_to(base_path)))
                    continue
                metrics = extract_metrics_from_log(log_file)
            else:
                # UniAD: 需要 JSON，用 converter 转成 VAD 格式的 summary 再解析
                if not json_file.exists():
                    # 兜底：有些实验的 json 文件名不严格等于 <log_stem>.json
                    # - 优先匹配 *rep_val*.json（且只有一个时才采用）
                    # - 否则如果 open_loop_eval 下只有一个 json，也采用
                    open_loop_dir = exp_folder / 'open_loop_eval'
                    rep_val_candidates = []
                    all_json_candidates = []
                    if open_loop_dir.exists():
                        all_json_candidates = sorted(open_loop_dir.glob("*.json"))
                        rep_val_candidates = [p for p in all_json_candidates if "rep_val" in p.name]
                    if len(rep_val_candidates) == 1:
                        json_file = rep_val_candidates[0]
                    elif len(all_json_candidates) == 1:
                        json_file = all_json_candidates[0]
                    else:
                        missing_log.append(str(exp_folder.relative_to(base_path)))
                        continue
                try:
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.json', delete=False, encoding='utf-8'
                    ) as f_out:
                        out_json = f_out.name
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.txt', delete=False, encoding='utf-8'
                    ) as f_sum:
                        summary_path = f_sum.name
                    ret = subprocess.run(
                        [
                            sys.executable,
                            str(converter_script),
                            str(json_file),
                            out_json,
                            '--summary',
                            summary_path,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if ret.returncode != 0:
                        extract_failed.append(str(exp_folder.relative_to(base_path)))
                        continue
                    metrics = extract_metrics_from_log(summary_path)
                    # 从 converter 输出的 JSON 统计 VAD 规则下发生碰撞的帧数（与 baseline 588 口径一致）
                    collision_count = count_collision_frames_vad_rule(out_json)
                    if metrics is not None and collision_count is not None:
                        metrics['collision_frame_count'] = collision_count
                    for p in (out_json, summary_path):
                        try:
                            os.unlink(p)
                        except OSError:
                            pass
                except Exception as e:
                    print(f"UniAD converter error for {exp_folder}: {e}")
                    extract_failed.append(str(exp_folder.relative_to(base_path)))
                    continue

            # 从路径中提取time horizon
            time_horizon = None
            folder_path_str = str(exp_folder)
            horizon_match = re.search(r'/([123])s/', folder_path_str)
            if horizon_match:
                time_horizon = f"{horizon_match.group(1)}s"
            
            # 如果指定了time_horizon，只处理匹配的
            if args.time_horizon and time_horizon != args.time_horizon:
                continue
            
            # 解析实验配置
            method, weight_count, fitness_type, repetition = parse_experiment_name(exp_folder.name)
            
            if not all([method, weight_count, fitness_type, repetition]):
                parse_failed.append(str(exp_folder.relative_to(base_path)))
                continue
            
            if metrics is None:
                extract_failed.append(str(exp_folder.relative_to(base_path)))
                continue
            
            # 存储time horizon数据（用于time horizon比较）
            if time_horizon:
                config_key = f"{method}_w{weight_count}_{fitness_type}"
                if config_key not in time_horizon_data:
                    time_horizon_data[config_key] = {}
                if repetition not in time_horizon_data[config_key]:
                    time_horizon_data[config_key][repetition] = {}
                time_horizon_data[config_key][repetition][time_horizon] = metrics
            
            # 存储数据
            data_entry = {
                'method': method,
                'weight_count': weight_count,
                'fitness_type': fitness_type,
                'repetition': repetition,
                'folder': exp_folder.name,
                **metrics
            }
            all_data.append(data_entry)
            success_count += 1
    
    # 详细报告数据提取情况
    print(f"\n" + "=" * 80)
    print("数据提取汇总")
    print("=" * 80)
    print(f"✓ 成功提取: {success_count} 个实验")
    print(f"✗ 缺少log文件: {len(missing_log)} 个实验")
    print(f"✗ 无法解析配置: {len(parse_failed)} 个实验")
    print(f"✗ 无法提取指标: {len(extract_failed)} 个实验")
    print(f"总计: {success_count + len(missing_log) + len(parse_failed) + len(extract_failed)} 个实验文件夹")
    
    if missing_log:
        print(f"\n缺少log文件的实验 ({len(missing_log)} 个):")
        for folder in missing_log[:20]:  # 只显示前20个
            print(f"  - {folder}")
        if len(missing_log) > 20:
            print(f"  ... 还有 {len(missing_log) - 20} 个")
    
    if parse_failed:
        print(f"\n无法解析配置的实验 ({len(parse_failed)} 个):")
        for folder in parse_failed[:20]:  # 只显示前20个
            print(f"  - {folder}")
        if len(parse_failed) > 20:
            print(f"  ... 还有 {len(parse_failed) - 20} 个")
    
    if extract_failed:
        print(f"\n无法提取指标的实验 ({len(extract_failed)} 个):")
        for folder in extract_failed[:20]:  # 只显示前20个
            print(f"  - {folder}")
        if len(extract_failed) > 20:
            print(f"  ... 还有 {len(extract_failed) - 20} 个")
    
    print(f"\n总共提取了 {len(all_data)} 个实验的数据")
    
    if len(all_data) == 0:
        print("\n" + "=" * 80)
        print("错误: 没有提取到任何实验数据！")
        print("=" * 80)
        print("\n请检查上述缺失数据的详细信息。")
        print("可能的原因：")
        print("1. 实验目录路径不正确")
        print("2. VAD: 缺少 open_loop_eval/<model>_rep_val.log；UniAD: 缺少 open_loop_eval/<model>_rep_val.json")
        print("3. log文件中不包含所需的指标数据（plan_L2_*s 或 plan_obj_box_col_*s）")
        print("4. 实验文件夹名称格式不正确，无法解析配置")
        print("\n脚本已停止执行。")
        sys.exit(1)
    
    # 转换为DataFrame便于分析
    df = pd.DataFrame(all_data)
    # 从数据中读取权重与 fitness 列表，支持 VAD (w26/52/105) 与 UniAD (w7/13/14 等)
    weight_counts = sorted(df['weight_count'].unique().astype(int).tolist())
    fitness_types = ['DISC', 'CONT', 'CONT2']
    methods = ['Arachne_v2', 'semSegRep']
    weight_labels = [f"w{w}" for w in weight_counts]
    
    # 定义要分析的指标（总是6个metric，不管time_horizon是什么）
    metrics_to_analyze = [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
    ]
    
    # 各配置下 L2 / 碰撞 均值统计
    print("\n" + "=" * 80)
    print("各配置下指标均值（L2 error 1s/2s/3s，碰撞率 1s/2s/3s）")
    print("=" * 80)
    config_cols = ['method', 'weight_count', 'fitness_type']
    mean_cols = ['plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
                 'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s']
    config_means = df.groupby(config_cols)[mean_cols].mean().round(6)
    print("\n【各配置均值】 (method, weight_count, fitness_type) -> 各指标均值")
    for idx in config_means.index:
        method, weight_count, fitness_type = idx
        row = config_means.loc[idx]
        n = int(df[(df['method'] == method) & (df['weight_count'] == weight_count) & (df['fitness_type'] == fitness_type)].shape[0])
        print(f"  {method}, w{weight_count}, {fitness_type} (n={n}): "
              f"L2_1s={row['plan_L2_1s']:.6f}, L2_2s={row['plan_L2_2s']:.6f}, L2_3s={row['plan_L2_3s']:.6f}, "
              f"col_1s={row['plan_obj_box_col_1s']:.6f}, col_2s={row['plan_obj_box_col_2s']:.6f}, col_3s={row['plan_obj_box_col_3s']:.6f}")
    # L2 3s 各配置平均值（单独列出）
    print("\n【L2 error 3s 各配置平均值】")
    l2_3s = df.groupby(config_cols)['plan_L2_3s'].agg(['mean', 'std', 'count']).round(6)
    for idx in l2_3s.index:
        method, weight_count, fitness_type = idx
        mean_val = l2_3s.at[idx, 'mean']
        std_val = l2_3s.at[idx, 'std']
        count_val = int(l2_3s.at[idx, 'count'])
        std_str = f"{std_val:.6f}" if pd.notna(std_val) else "N/A"
        print(f"  {method}, w{weight_count}, {fitness_type}: mean={mean_val:.6f}, std={std_str}, n={count_val}")
    # Collision 3s 各配置平均值（单独列出）；UniAD 为发生碰撞的帧数，VAD 为碰撞率
    print("\n【Collision 3s 各配置平均值】")
    collision_col = 'collision_frame_count' if args.model_type == 'uniad' else 'plan_obj_box_col_3s'
    col_3s = df.groupby(config_cols)[collision_col].agg(['mean', 'std', 'count']).round(6)
    for idx in col_3s.index:
        method, weight_count, fitness_type = idx
        mean_val = col_3s.at[idx, 'mean']
        std_val = col_3s.at[idx, 'std']
        count_val = int(col_3s.at[idx, 'count'])
        std_str = f"{std_val:.6f}" if pd.notna(std_val) else "N/A"
        mean_fmt = f"{mean_val:.0f}" if args.model_type == 'uniad' else f"{mean_val:.6f}"
        print(f"  {method}, w{weight_count}, {fitness_type}: mean={mean_fmt}, std={std_str}, n={count_val}")
    print("=" * 80)

    # L2 3s 与 Collision 3s 各配置 10 次重复的箱线图，并标出“初始未修复”的基线值
    # VAD 与 UniAD 使用不同的 baseline（均按有效帧统计）
    if args.model_type == 'vad':
        # VAD baseline（来自原始 VAD open-loop 评估结果）
        # 有效帧 6121，碰撞次数 ≈ BASELINE_COLLISION_3S × 6121 × 6 ≈ 85
        BASELINE_L2_3S = 1.4262655224607823
        BASELINE_COLLISION_3S = 0.002314436707869014
    else:
        # UniAD baseline：来自 uniad_base_baseline_b2d_infos_val_partB_25clips.json 经
        # baseline/convert_uniad_to_vad_metrics.py 转成 VAD 规则后的 summary：
        # - 有效帧 = fut_valid_flag（VAD 规则：full_future and mask_sum > 0），n=6371
        # - BASELINE_L2_3S = plan_L2_3s (avg_valid)
        # - BASELINE_COLLISION_3S = plan_obj_box_col_3s (avg_valid)，即有效帧上每帧
        #   plan_obj_box_col_1s/2s/3s 非负均值再整体平均
        # - BASELINE_COLLISION_FRAME_COUNT = VAD 规则下发生碰撞的帧数（图例用）
        BASELINE_L2_3S = 1.2513238752833544
        BASELINE_COLLISION_3S = 0.014457873542816614
        BASELINE_COLLISION_FRAME_COUNT = 588
    if HAS_MATPLOTLIB:
        # 为 boxplot 明确指定顺序：
        #  method 固定顺序: semSegRep（前9个）-> Arachne_v2（后9个）
        #  在每个 method 下，先按 fitness_type: CONT -> CONT2 -> DISC
        #  再按 weight_count 升序: w7 -> w13 -> w26 (或其他实际权重)
        config_order = []
        for method in ['semSegRep', 'Arachne_v2']:
            # 注意：这里显式指定 boxplot 中 fitness_type 的顺序：
            #  先 CONT，再 CONT2，最后 DISC
            for fitness_type in ['CONT', 'CONT2', 'DISC']:
                for weight_count in weight_counts:
                    key = (method, weight_count, fitness_type)
                    if key in config_means.index:
                        config_order.append(key)
        short_labels = []
        # 将 (method, fitness_type, weight_count) 映射到 LaTeX/mathtext 友好的标签：
        #  形如 "$FL_{adv}-fit_{disc}-Low$"（无括号、无逗号，用短横线连接）
        loss_map = {
            "Arachne_v2": r"FL_{adv}",
            "semSegRep": r"FL_{naive}",
        }
        fit_map = {
            "DISC": r"fit_{CA}",
            "CONT": r"fit_{ER}",
            "CONT2": r"fit_{ERC}",
        }
        # 权重等级映射：
        # - UniAD: w7 -> Low, w13 -> Mid, w26 -> High
        # - VAD:   w26 -> Low, w52 -> Mid, w105 -> High
        if args.model_type == 'uniad':
            level_map = {
                7: "Low",
                13: "Med",
                26: "High",
            }
        else:
            level_map = {
                26: "Low",
                52: "Med",
                105: "High",
            }

        for (method, weight_count, fitness_type) in config_order:
            loss_str = loss_map.get(method, method)
            fit_str = fit_map.get(fitness_type, fitness_type)
            try:
                w_int = int(weight_count)
            except Exception:
                w_int = None
            level_str = level_map.get(w_int, f"w{weight_count}")
            # 使用 mathtext 语法渲染 LaTeX 风格标签，三部分用短横线连接
            label = rf"${loss_str}-{fit_str}-{level_str}$"
            short_labels.append(label)
        l2_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['plan_L2_3s'].dropna().values
                   for (m, w, f) in config_order]
        # VAD：collision 为 plan_obj_box_col_3s (率) × 有效帧×6 → 碰撞次数；UniAD：每实验 JSON 统计的发生碰撞的帧数
        VAD_VALID_FRAMES = 6121
        VAD_COLLISION_SCALE = VAD_VALID_FRAMES * 6
        if args.model_type == 'vad':
            col_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['plan_obj_box_col_3s'].dropna().values * VAD_COLLISION_SCALE
                       for (m, w, f) in config_order]
            baseline_collision_plot = BASELINE_COLLISION_3S * VAD_COLLISION_SCALE
        else:
            col_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['collision_frame_count'].dropna().values
                       for (m, w, f) in config_order]
            baseline_collision_plot = BASELINE_COLLISION_FRAME_COUNT
        out_dir = Path(__file__).parent
        # 两图统一尺寸和边距，便于上下对齐；高度适中便于放论文
        FIG_SIZE = (14, 5)
        SUBPLOT_LEFT, SUBPLOT_RIGHT = 0.06, 0.98
        SUBPLOT_BOTTOM, SUBPLOT_TOP = 0.22, 0.90
        # 全局字体大小（轴标签、刻度、图例等），便于论文排版
        plt.rcParams['font.size'] = 18
        # L2 error 3s 一张图（18 个配置）+ 初始值水平线
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        fig1.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP)
        bp1 = ax1.boxplot(l2_data, labels=short_labels, patch_artist=True, showfliers=False)
        ax1.axhline(
            y=BASELINE_L2_3S,
            color='red',
            linestyle='--',
            linewidth=1.5,
            # L2_Err(M_{orig}, D_{test})
            label=rf"$\mu_{{L2\_Err}}(M_{{orig}}, D_{{test}})$ = {BASELINE_L2_3S:.4f}",
        )
        ax1.set_ylabel('L2 error')
        ax1.tick_params(axis='x', rotation=45)
        # 让每个 x 轴标签的右端点对齐刻度，避免长标签视觉错位
        for label in ax1.get_xticklabels():
            label.set_ha('right')
        ax1.legend(loc='upper right')
        # 更柔和的配色（类似 seaborn 风格），并用深色中位数线避免被盖住
        for patch in bp1['boxes']:
            patch.set_facecolor('#8da0cb')  # 柔和蓝紫
        for median in bp1['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        fig1.savefig(out_dir / 'l2_3s_boxplot.pdf', bbox_inches='tight')
        plt.close(fig1)
        # Collision 3s 一张图（18 个配置）+ 初始值水平线（VAD 为碰撞次数，UniAD 为发生碰撞的帧数）
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        fig2.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP)
        bp2 = ax2.boxplot(col_data, labels=short_labels, patch_artist=True, showfliers=False)
        ax2.axhline(
            y=baseline_collision_plot,
            color='red',
            linestyle='--',
            linewidth=1.5,
            # 混合写法：前面的 '#' 用普通文本，后面的 colls(...) 用 LaTeX
            label="#" + rf"$\mathrm{{colls}}(M_{{orig}}, D_{{test}})$ = {baseline_collision_plot:.0f}",
        )
        ax2.set_ylabel('#colls')
        ax2.tick_params(axis='x', rotation=45)
        for label in ax2.get_xticklabels():
            label.set_ha('right')
        ax2.legend(loc='upper right')
        for patch in bp2['boxes']:
            patch.set_facecolor('#fc8d62')  # 柔和橙色
        for median in bp2['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        fig2.savefig(out_dir / 'collision_3s_boxplot.pdf', bbox_inches='tight')
        plt.close(fig2)
        print(f"\n箱线图已保存: {out_dir / 'l2_3s_boxplot.pdf'}, {out_dir / 'collision_3s_boxplot.pdf'}")
    else:
        print("\n(未安装 matplotlib，跳过箱线图)")

    print("\n" + "=" * 80)
    print("Step 2: 计算Cohen's d统计量")
    print("=" * 80)
    
    results = []
    
    # 比较1: 不同方法之间的比较 (Arachne_v2 vs semSegRep)
    # 使用配对比较：先计算每种配置的均值，然后配对比较
    print("\n--- 比较1: 方法比较 (Arachne_v2 vs semSegRep) ---")
    for metric in metrics_to_analyze:
        method1, method2 = 'Arachne_v2', 'semSegRep'
        if method1 not in df['method'].unique() or method2 not in df['method'].unique():
            continue

        configs = []
        m1_means = []
        m2_means = []
        for weight_count in weight_counts:
            for fitness_type in fitness_types:
                v1 = df[(df['method'] == method1) &
                        (df['fitness_type'] == fitness_type) &
                        (df['weight_count'] == weight_count)][metric].dropna().values
                v2 = df[(df['method'] == method2) &
                        (df['fitness_type'] == fitness_type) &
                        (df['weight_count'] == weight_count)][metric].dropna().values
                if len(v1) > 0 and len(v2) > 0:
                    configs.append((weight_count, fitness_type))
                    m1_means.append(float(np.mean(v1)))
                    m2_means.append(float(np.mean(v2)))

        if len(m1_means) > 0 and len(m2_means) == len(m1_means):
            m1_array = np.asarray(m1_means, dtype=np.float64)
            m2_array = np.asarray(m2_means, dtype=np.float64)
            d = cohens_d(m1_array, m2_array, paired=True)

            results.append({
                'comparison_type': 'Method',
                'group1': method1,
                'group2': method2,
                'fitness_type': None,
                'weight_count': None,
                'metric': metric,
                'cohens_d': d,
                'mean1': float(np.mean(m1_array)),
                'mean2': float(np.mean(m2_array)),
                'std1': float(np.std(m1_array, ddof=1)) if len(m1_array) > 1 else 0.0,
                'std2': float(np.std(m2_array, ddof=1)) if len(m2_array) > 1 else 0.0,
                'n1': int(len(m1_array)),
                'n2': int(len(m2_array))
            })

            if d is not None:
                order_str = ", ".join([f"(w{w}, {ft})" for (w, ft) in configs])
                print(f"\n  {metric}:")
                print(f"    向量维度: {len(m1_array)}×1；维度顺序: {order_str}")
                print(f"    {method1}向量: {m1_array}")
                print(f"    {method2}向量: {m2_array}")
    
    # 比较2: 不同fitness函数之间的比较
    print("\n--- 比较2: Fitness函数比较 ---")
    for metric in metrics_to_analyze:
        # 构造“配对维度”：每个维度是一对 (method, weight_count)，要求三个fitness在该维度都有数据
        base_positions = []
        for method in methods:
            if method not in df['method'].unique():
                continue
            for weight_count in weight_counts:
                ok = True
                for ft in fitness_types:
                    values = df[(df['method'] == method) &
                                (df['fitness_type'] == ft) &
                                (df['weight_count'] == weight_count)][metric].dropna().values
                    if len(values) == 0:
                        ok = False
                        break
                if ok:
                    base_positions.append((method, weight_count))

        fitness_vectors = {}
        for ft in fitness_types:
            vec = []
            for method, weight_count in base_positions:
                values = df[(df['method'] == method) &
                            (df['fitness_type'] == ft) &
                            (df['weight_count'] == weight_count)][metric].dropna().values
                vec.append(float(np.mean(values)))
            if len(vec) > 0:
                fitness_vectors[ft] = np.asarray(vec, dtype=np.float64)

        if len(base_positions) > 0 and len(fitness_vectors) == len(fitness_types):
            print(f"\n  {metric}:")
            order_str = ", ".join([f"({m}, w{w})" for (m, w) in base_positions])
            print(f"    向量维度: {len(base_positions)}×1；维度顺序: {order_str}")
            for ft in fitness_types:
                print(f"    {ft}向量: {fitness_vectors[ft]}")

            for i, fitness1 in enumerate(fitness_types):
                for j, fitness2 in enumerate(fitness_types):
                    if i == j:
                        continue
                    vec1 = fitness_vectors[fitness1]
                    vec2 = fitness_vectors[fitness2]
                    d = cohens_d(vec1, vec2, paired=True)
                    if d is not None:
                        results.append({
                            'comparison_type': 'Fitness',
                            'group1': fitness1,
                            'group2': fitness2,
                            'method': None,
                            'weight_count': None,
                            'metric': metric,
                            'cohens_d': d,
                            'mean1': float(np.mean(vec1)),
                            'mean2': float(np.mean(vec2)),
                            'std1': float(np.std(vec1, ddof=1)) if len(vec1) > 1 else 0.0,
                            'std2': float(np.std(vec2, ddof=1)) if len(vec2) > 1 else 0.0,
                            'n1': int(len(vec1)),
                            'n2': int(len(vec2))
                        })

    # 比较3: 不同权重数量之间的比较
    print("\n--- 比较3: 权重数量比较 ---")
    for metric in metrics_to_analyze:
        # 构造“配对维度”：每个维度是一对 (method, fitness_type)，要求所有 weight_counts 在该维度都有数据
        base_positions = []
        for method in methods:
            if method not in df['method'].unique():
                continue
            for ft in fitness_types:
                ok = True
                for w in weight_counts:
                    values = df[(df['method'] == method) &
                                (df['fitness_type'] == ft) &
                                (df['weight_count'] == w)][metric].dropna().values
                    if len(values) == 0:
                        ok = False
                        break
                if ok:
                    base_positions.append((method, ft))

        weight_vectors = {}
        for w, label in zip(weight_counts, weight_labels):
            vec = []
            for method, ft in base_positions:
                values = df[(df['method'] == method) &
                            (df['fitness_type'] == ft) &
                            (df['weight_count'] == w)][metric].dropna().values
                vec.append(float(np.mean(values)))
            if len(vec) > 0:
                weight_vectors[label] = np.asarray(vec, dtype=np.float64)

        if len(base_positions) > 0 and len(weight_vectors) == len(weight_counts):
            print(f"\n  {metric}:")
            order_str = ", ".join([f"({m}, {ft})" for (m, ft) in base_positions])
            print(f"    向量维度: {len(base_positions)}×1；维度顺序: {order_str}")
            for label in weight_labels:
                if label in weight_vectors:
                    print(f"    {label}向量: {weight_vectors[label]}")

            for i, w1 in enumerate(weight_labels):
                for j, w2 in enumerate(weight_labels):
                    if i == j:
                        continue
                    vec1 = weight_vectors.get(w1)
                    vec2 = weight_vectors.get(w2)
                    if vec1 is None or vec2 is None:
                        continue
                    d = cohens_d(vec1, vec2, paired=True)
                    if d is not None:
                        results.append({
                            'comparison_type': 'Weight',
                            'group1': w1,
                            'group2': w2,
                            'method': None,
                            'fitness_type': None,
                            'metric': metric,
                            'cohens_d': d,
                            'mean1': float(np.mean(vec1)),
                            'mean2': float(np.mean(vec2)),
                            'std1': float(np.std(vec1, ddof=1)) if len(vec1) > 1 else 0.0,
                            'std2': float(np.std(vec2, ddof=1)) if len(vec2) > 1 else 0.0,
                            'n1': int(len(vec1)),
                            'n2': int(len(vec2))
                        })

    # 转换为DataFrame便于汇总
    results_df = pd.DataFrame(results)
    
    # 打印汇总报告
    print("\n" + "=" * 80)
    print("Step 3: 汇总报告")
    print("=" * 80)
    
    # 按比较类型和指标汇总
    print("\nCohen's d 统计分析结果汇总\n")

    for comparison_type in ['Method', 'Fitness', 'Weight']:
        print(f"\n{'=' * 80}")
        print(f"比较类型: {comparison_type}")
        print(f"{'=' * 80}\n")

        subset = results_df[results_df['comparison_type'] == comparison_type]

        for metric in metrics_to_analyze:
            metric_subset = subset[subset['metric'] == metric]
            if len(metric_subset) > 0:
                if comparison_type == 'Fitness':
                    # Fitness比较：显示3×3矩阵
                    fitness_types = ['DISC', 'CONT', 'CONT2']
                    # 构建矩阵
                    matrix = {}
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            effect_size = interpret_cohens_d(d)
                            key = (row['group1'], row['group2'])
                            matrix[key] = effect_size
                    
                    print(f"指标：{metric}；DISC vs CONT vs CONT2")
                    print("    " + " " * 12 + "DISC" + " " * 12 + "CONT" + " " * 12 + "CONT2")
                    for i, fitness1 in enumerate(fitness_types):
                        row_str = f"    {fitness1:12s}"
                        for j, fitness2 in enumerate(fitness_types):
                            if i == j:
                                row_str += "equivalent".center(16)
                            else:
                                key = (fitness1, fitness2)
                                if key in matrix:
                                    row_str += matrix[key].center(16)
                                else:
                                    row_str += "identical".center(16)  # N/A表示数据完全相同或差异完全相同
                        print(row_str)
                elif comparison_type == 'Weight':
                    # Weight比较：显示 N×N 矩阵（N = 实际权重数量）
                    # 优先按 weight_counts 排序，其次按字典序兜底
                    weight_names = list(weight_labels)
                    if not weight_names:
                        # 从结果里兜底提取
                        names = set(metric_subset['group1'].dropna().tolist()) | set(metric_subset['group2'].dropna().tolist())
                        def _w_key(x):
                            m = re.search(r'(\d+)', str(x))
                            return int(m.group(1)) if m else 10**9
                        weight_names = sorted(names, key=_w_key)
                    # 构建矩阵
                    matrix = {}
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            effect_size = interpret_cohens_d(d)
                            key = (row['group1'], row['group2'])
                            matrix[key] = effect_size
                    
                    header = "    " + " " * 12 + "".join([f"{w:>16s}" for w in weight_names])
                    print(f"指标：{metric}；" + " vs ".join(weight_names))
                    print(header)
                    for i, weight1 in enumerate(weight_names):
                        row_str = f"    {weight1:12s}"
                        for j, weight2 in enumerate(weight_names):
                            if i == j:
                                row_str += "equivalent".center(16)
                            else:
                                key = (weight1, weight2)
                                if key in matrix:
                                    row_str += matrix[key].center(16)
                                else:
                                    row_str += "identical".center(16)
                        print(row_str)
                else:
                    # Method比较：一行显示
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            # 使用新规则解释效应量大小
                            effect_size = interpret_cohens_d(d)
                            
                            print(f"指标：{metric}；{row['group1']} vs {row['group2']}：{effect_size}")
    
    # Time Horizon比较
    if args.compare_horizons:
        print("\n" + "=" * 80)
        print("Step 4: 计算Time Horizon比较（1s vs 2s vs 3s）")
        print("=" * 80)
        print("比较不同time horizon目录下的实验结果")
        print("每个time horizon都会产生6个metric，所以需要进行6次比较")
        
        # 定义所有6个metric（每个time horizon都会产生这6个metric）
        metrics_to_compare = [
            'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
            'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
        ]
        time_horizons = ['1s', '2s', '3s']
        
        # 生成所有配置的列表（按固定顺序：方法 -> 权重 -> fitness），权重来自实际数据
        configs = []
        for method in methods:
            for weight_count in weight_counts:
                for fitness_type in fitness_types:
                    configs.append(f"{method}_w{weight_count}_{fitness_type}")
        
        # 对每个metric进行比较
        for metric_name in metrics_to_compare:
            print(f"\n指标: {metric_name}")
            print("=" * 80)
            
            # 为每个time horizon构建 N 维向量（每个维度是一个配置的均值）
            horizon_vectors = {th: [] for th in time_horizons}
            
            # 对每个配置，计算该time horizon下所有重复实验的均值
            # 统计信息：记录每个time horizon下每个配置的重复实验数量
            repetition_counts = {th: [] for th in time_horizons}
            
            # 记录每个配置在每个time horizon下有哪些重复实验
            config_repetition_details = {config_name: {th: [] for th in time_horizons} for config_name in configs}
            
            for config_name in configs:
                for th in time_horizons:
                    config_values = []
                    found_repetitions = []
                    if config_name in time_horizon_data:
                        for repetition in time_horizon_data[config_name]:
                            rep_data = time_horizon_data[config_name][repetition]
                            if th in rep_data:
                                # 从该time horizon的数据中提取对应的metric
                                if metric_name in rep_data[th] and rep_data[th][metric_name] is not None:
                                    config_values.append(rep_data[th][metric_name])
                                    found_repetitions.append(repetition)
                    
                    # 记录该配置在该time horizon下找到的重复实验
                    config_repetition_details[config_name][th] = sorted(found_repetitions)
                    
                    # 记录该配置在该time horizon下的重复实验数量
                    repetition_counts[th].append(len(config_values))
                    
                    # 计算该配置在该time horizon下的均值
                    if len(config_values) > 0:
                        horizon_vectors[th].append(np.mean(config_values))
                    else:
                        horizon_vectors[th].append(np.nan)
            
            # 显示统计信息：确认每个time horizon的重复实验数量
            print(f"\n  重复实验数量统计（每个配置）：")
            for th in time_horizons:
                counts = repetition_counts[th]
                if len(counts) > 0:
                    unique_counts = np.unique(counts)
                    # 统计每个数量出现的次数
                    count_freq = {}
                    for c in counts:
                        count_freq[c] = count_freq.get(c, 0) + 1
                    freq_str = ", ".join([f"{v}个配置有{k}次" for k, v in sorted(count_freq.items())])
                    exp_hint = ""
                    max_count = int(np.max(counts)) if len(counts) > 0 else 0
                    if max_count > 0:
                        exp_hint = f"（该目录下最大重复数={max_count}）"
                    print(f"    {th}: 唯一值={unique_counts}, 分布: {freq_str} {exp_hint}")
                    
                    # 显示缺少数据的配置
                    missing_configs = []
                    for i, count in enumerate(counts):
                        expected = max_count
                        if expected > 0 and count < expected:
                            missing_configs.append((configs[i], count, expected))
                    
                    if missing_configs:
                        print(f"    缺少数据的配置（{th}目录下，metric={metric_name}）:")
                        for config, actual, expected in missing_configs:
                            found_reps = config_repetition_details[config][th]
                            expected_reps = list(range(1, expected + 1))
                            missing_reps = [r for r in expected_reps if r not in found_reps]
                            if missing_reps and expected > 0:
                                print(f"      - {config}: 只有{actual}个（找到: {found_reps}），缺少: {missing_reps}，参考最大重复数={expected}个")
                            else:
                                print(f"      - {config}: 只有{actual}个（找到: {found_reps}），参考最大重复数={expected}个")
            
            # 转换为numpy数组（18维向量）
            for th in time_horizons:
                horizon_vectors[th] = np.array(horizon_vectors[th])
            
            # 计算配对Cohen's d（18维向量之间的比较）
            print(f"\n  {metric_name}:")
            print(f"    1s向量 (18×1): {horizon_vectors['1s']}")
            print(f"    2s向量 (18×1): {horizon_vectors['2s']}")
            print(f"    3s向量 (18×1): {horizon_vectors['3s']}")
            print(f"\n  注意：使用配对Cohen's d比较{len(configs)}维向量（每个维度对应一个配置的均值）")
            
            # 构建3×3矩阵
            matrix = {}
            for i, th1 in enumerate(time_horizons):
                for j, th2 in enumerate(time_horizons):
                    if i != j:
                        vec1 = horizon_vectors[th1]
                        vec2 = horizon_vectors[th2]
                        # 移除NaN值（如果有配置缺少数据）
                        valid_mask = ~(np.isnan(vec1) | np.isnan(vec2))
                        vec1_valid = vec1[valid_mask]
                        vec2_valid = vec2[valid_mask]
                        
                        if len(vec1_valid) > 0 and len(vec2_valid) > 0 and len(vec1_valid) == len(vec2_valid):
                            # 使用配对Cohen's d（paired=True），因为这是18维向量之间的比较
                            d = cohens_d(vec1_valid, vec2_valid, paired=True)
                            if d is not None:
                                matrix[(th1, th2)] = d
                                effect_size = interpret_cohens_d(d)
                                print(f"    {th1} vs {th2}: Cohen's d = {d:.4f}, {effect_size} (有效维度: {len(vec1_valid)}/{len(configs)})")
            
            # 显示3×3矩阵
            print(f"\n  {metric_name} Time Horizon比较矩阵:")
            print("    " + " " * 12 + "1s" + " " * 12 + "2s" + " " * 12 + "3s")
            for i, th1 in enumerate(time_horizons):
                row_str = f"    {th1:12s}"
                for j, th2 in enumerate(time_horizons):
                    if i == j:
                        row_str += "equivalent".center(16)
                    else:
                        key = (th1, th2)
                        if key in matrix:
                            effect_size = interpret_cohens_d(matrix[key])
                            row_str += effect_size.center(16)
                        else:
                            row_str += "N/A".center(16)
                print(row_str)
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

