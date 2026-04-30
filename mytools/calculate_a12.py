#!/usr/bin/env python3
"""
Compute Vargha and Delaney's Aˆ12 effect size.

Aˆ12 is a non-parametric effect size for comparing two independent (unpaired) samples.
Aˆ12 = P(X1 > X2) + 0.5 * P(X1 = X2)

When to use:
- Two independent groups
- Each group has multiple observations (e.g. all values from repeated runs)
- All repetition values enter the computation; no need to average first

Range: [0, 1]
- Aˆ12 = 0.5: no difference between samples
- Aˆ12 > 0.5: first sample tends to be larger than the second
- Aˆ12 < 0.5: first sample tends to be smaller than the second

Effect size labels (Vargha & Delaney, 2000):
- negligible: Aˆ12 ∈ (0.5, 0.556) or Aˆ12 ∈ (0.444, 0.5)
- small: Aˆ12 ∈ [0.556, 0.638) or Aˆ12 ∈ (0.362, 0.444]
- medium: Aˆ12 ∈ [0.638, 0.714) or Aˆ12 ∈ (0.286, 0.362]
- large: Aˆ12 ≥ 0.714 or Aˆ12 ≤ 0.286

Usage:
conda activate b2d_zoo
python calculate_a12.py
"""

import os
import re
import sys
import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
except ImportError:
    print("Error: numpy, pandas and scipy are required. Please activate b2d_zoo conda environment:")
    print("  conda activate b2d_zoo")
    sys.exit(1)

try:
    import openpyxl
except ImportError:
    print("Warning: openpyxl is not installed. Excel file generation will fail.")
    print("  Please install it: pip install openpyxl")


def calculate_a12(group1, group2):
    """
    Compute Vargha and Delaney's Aˆ12 effect size.

    Parameters:
    -----------
    group1 : array-like
        Data for the first sample
    group2 : array-like
        Data for the second sample

    Returns:
    --------
    a12 : float
        Aˆ12 in [0, 1]
        - a12 = 0.5: no difference
        - a12 > 0.5: group1 tends to be larger than group2
        - a12 < 0.5: group1 tends to be smaller than group2
    """
    group1 = np.array(group1)
    group2 = np.array(group2)

    if len(group1) == 0 or len(group2) == 0:
        return None

    n1 = len(group1)
    n2 = len(group2)

    # Method 1: rank-sum formulation (efficient)
    # Merge samples and assign ranks
    combined = np.concatenate([group1, group2])
    ranks = stats.rankdata(combined, method='average')  # Average ranks for ties

    # Rank sum of the first sample
    r1 = np.sum(ranks[:n1])

    # Aˆ12 = (R1 - n1(n1+1)/2) / (n1 * n2)
    a12 = (r1 - n1 * (n1 + 1) / 2) / (n1 * n2)

    return a12

def _get_metric_values(all_data, config_name, metric):
    """Safely return all repetition values for a config and metric, or None."""
    if config_name not in all_data:
        return None
    if metric not in all_data[config_name]:
        return None
    values = all_data[config_name][metric]
    if not values:
        return None
    return values


def interpret_a12(a12):
    """
    Map Aˆ12 to Vargha & Delaney (2000) effect size labels.

    Thresholds:
    - negligible: Aˆ12 ∈ (0.5, 0.556) or Aˆ12 ∈ (0.444, 0.5)
    - small: Aˆ12 ∈ [0.556, 0.638) or Aˆ12 ∈ (0.362, 0.444]
    - medium: Aˆ12 ∈ [0.638, 0.714) or Aˆ12 ∈ (0.286, 0.362]
    - large: Aˆ12 ≥ 0.714 or Aˆ12 ≤ 0.286

    Parameters:
    -----------
    a12 : float
        Aˆ12 value

    Returns:
    --------
    interpretation : str
        Effect size label
    """
    if a12 is None:
        return "N/A"

    if a12 == 0.5:
        return "negligible"
    elif a12 > 0.5:
        if a12 < 0.556:
            return "negligible"
        elif a12 < 0.638:
            return "small"
        elif a12 < 0.714:
            return "medium"
        else:  # a12 >= 0.714
            return "large"
    else:  # a12 < 0.5
        if a12 > 0.444:
            return "negligible"
        elif a12 > 0.362:
            return "small"
        elif a12 > 0.286:
            return "medium"
        else:  # a12 <= 0.286
            return "large"


def interpret_a12_directional(a12, row_config=None, col_config=None):
    """
    Directional Aˆ12 labels for an 18×18 table.

    Rows are row_config; columns are col_config. Metrics are lower-is-better.
    - Aˆ12 > 0.5: row tends to be larger → row is worse (returns *_worse)
    - Aˆ12 < 0.5: row tends to be smaller → row is better (returns *_better)

    Thresholds per Vargha & Delaney (2000):
    - negligible: Aˆ12 ∈ (0.5, 0.556) or Aˆ12 ∈ (0.444, 0.5)
    - small: Aˆ12 ∈ [0.556, 0.638) or Aˆ12 ∈ (0.362, 0.444]
    - medium: Aˆ12 ∈ [0.638, 0.714) or Aˆ12 ∈ (0.286, 0.362]
    - large: Aˆ12 ≥ 0.714 or Aˆ12 ≤ 0.286

    Parameters:
    -----------
    a12 : float
        Aˆ12 value
    row_config : str, optional
        Row configuration name (for clearer messages)
    col_config : str, optional
        Column configuration name

    Returns:
    --------
    interpretation : str
        Directional effect label
    """
    if a12 is None:
        return "N/A"

    if a12 == 0.5:
        return "equivalent"
    elif a12 > 0.5:
        # Aˆ12 > 0.5: row values tend to exceed column values
        # For lower-is-better metrics, row is worse and column is better
        if a12 < 0.556:
            return "negligible_worse"
        elif a12 < 0.638:
            return "small_worse"
        elif a12 < 0.714:
            return "medium_worse"
        else:  # a12 >= 0.714
            return "large_worse"
    else:  # a12 < 0.5
        # Aˆ12 < 0.5: row values tend to be below column values
        # For lower-is-better metrics, row is better and column is worse
        if a12 > 0.444:
            return "negligible_better"
        elif a12 > 0.362:
            return "small_better"
        elif a12 > 0.286:
            return "medium_better"
        else:  # a12 <= 0.286
            return "large_better"


def extract_metrics_from_log(log_file):
    """Parse metrics from a log file."""
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

            # L2 errors
            for horizon in [1, 2, 3]:
                pattern = rf'plan_L2_{horizon}s:([\d.e+-]+)'
                match = re.search(pattern, content)
                if match:
                    metrics[f'plan_L2_{horizon}s'] = float(match.group(1))
                else:
                    metrics[f'plan_L2_{horizon}s'] = None

            # Collision rates
            for horizon in [1, 2, 3]:
                pattern = rf'plan_obj_box_col_{horizon}s:([\d.e+-]+)'
                match = re.search(pattern, content)
                if match:
                    metrics[f'plan_obj_box_col_{horizon}s'] = float(match.group(1))
                else:
                    metrics[f'plan_obj_box_col_{horizon}s'] = None

    except Exception as e:
        print(f"Error reading {log_file}: {e}")
        return None

    return metrics if metrics else None


def parse_experiment_name(folder_name):
    """
    Parse experiment folder name.
    Pattern: VAD_base_REP_VAL_{method}_DE_w{weight_count}_p*_i*_es*_{fitness_type}_{repetition}
    """
    method = None
    weight_count = None
    fitness_type = None
    repetition = None

    # Method name
    if 'Arachne_v2' in folder_name:
        method = 'Arachne_v2'
    elif 'semSegRep' in folder_name:
        method = 'semSegRep'

    # Weight count
    weight_match = re.search(r'_w(\d+)_', folder_name)
    if weight_match:
        weight_count = int(weight_match.group(1))

    # Fitness type
    if '_DISC_' in folder_name:
        fitness_type = 'DISC'
    elif '_CONT2_' in folder_name:
        fitness_type = 'CONT2'
    elif '_CONT_' in folder_name:
        fitness_type = 'CONT'

    # Repetition index
    rep_match = re.search(r'_(CONT|CONT2|DISC)_(\d+)$', folder_name)
    if rep_match:
        repetition = int(rep_match.group(2))

    return method, weight_count, fitness_type, repetition


def generate_config_name(method, weight_count, fitness_type):
    """Build a configuration string."""
    return f"{method}_w{weight_count}_{fitness_type}"

def generate_group_name(method, fitness_type):
    """Build group label (method + fitness) for the 6×6 table."""
    return f"{method}_{fitness_type}"

def _metric_short_name(metric: str) -> str:
    """Shorten metric names to fit Excel sheet name limits."""
    mapping = {
        'plan_L2_1s': 'L2_1s',
        'plan_L2_2s': 'L2_2s',
        'plan_L2_3s': 'L2_3s',
        'plan_obj_box_col_1s': 'COL_1s',
        'plan_obj_box_col_2s': 'COL_2s',
        'plan_obj_box_col_3s': 'COL_3s',
    }
    return mapping.get(metric, metric)

def _make_unique_sheet_name(base: str, used: set) -> str:
    """
    Excel sheet names are limited to 31 characters.
    Truncate to <=31 and append _1, _2, ... on collisions.
    """
    base = base[:31]
    if base not in used:
        used.add(base)
        return base
    i = 1
    while True:
        suffix = f"_{i}"
        candidate = (base[:31 - len(suffix)] + suffix)[:31]
        if candidate not in used:
            used.add(candidate)
            return candidate
        i += 1


def main():
    """Compute pairwise Aˆ12 for all 18 configurations."""
    # CLI
    parser = argparse.ArgumentParser(description="Compute Aˆ12 statistics")
    parser.add_argument('--time_horizon', type=str, choices=['1s', '2s', '3s'], default=None,
                        help='Only read runs under this time-horizon subdir (default: all)')
    parser.add_argument('--compare_horizons', action='store_true',
                        help='Compare across time horizons (1s vs 2s vs 3s)')
    parser.add_argument('--self_check', action='store_true',
                        help='Consistency check: A12(X,Y)+A12(Y,X)≈1 and diagonal 0.5')
    args = parser.parse_args()

    # Experiment roots
    base_dirs = [
        '/home/deyun/git/B2DRepair/vad_base_Arachne_v2_DE_results',
        '/home/deyun/git/B2DRepair/vad_base_semSegRep_DE_results'
    ]

    # config -> metric -> list of values from all repetitions
    all_data = {}
    # config -> repetition -> time_horizon -> metric -> value
    time_horizon_data = {}

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

        # If time_horizon is set, search only under that subdirectory
        if args.time_horizon:
            search_path = base_path / args.time_horizon
            if not search_path.exists():
                print(f"  Warning: {search_path} not found")
                continue
            exp_folders = []
            for exp_folder in search_path.rglob('VAD_base_REP_VAL_*'):
                if exp_folder.is_dir():
                    exp_folders.append(exp_folder)
        else:
            # Recursively find folders starting with VAD_base_REP_VAL_
            exp_folders = []
            for exp_folder in base_path.rglob('VAD_base_REP_VAL_*'):
                if exp_folder.is_dir():
                    exp_folders.append(exp_folder)

        exp_folders = sorted(exp_folders)
        print(f"  找到 {len(exp_folders)} 个实验文件夹")

        for exp_folder in exp_folders:
            log_file = exp_folder / 'open_loop_eval' / 'vad_base_rep_val.log'

            if not log_file.exists():
                continue

            # Infer time horizon from path
            time_horizon = None
            folder_path_str = str(exp_folder)
            horizon_match = re.search(r'/([123])s/', folder_path_str)
            if horizon_match:
                time_horizon = f"{horizon_match.group(1)}s"

            # Skip if user requested a specific horizon and this folder differs
            if args.time_horizon and time_horizon != args.time_horizon:
                continue

            # Parse experiment metadata
            method, weight_count, fitness_type, repetition = parse_experiment_name(exp_folder.name)

            if not all([method, weight_count, fitness_type, repetition]):
                continue

            # Metrics from log
            metrics = extract_metrics_from_log(log_file)

            if metrics is None:
                continue

            # Config key
            config_name = generate_config_name(method, weight_count, fitness_type)

            # Store per-horizon data for optional horizon comparison
            if time_horizon:
                if config_name not in time_horizon_data:
                    time_horizon_data[config_name] = {}
                if repetition not in time_horizon_data[config_name]:
                    time_horizon_data[config_name][repetition] = {}
                time_horizon_data[config_name][repetition][time_horizon] = metrics

            # Initialize config bucket
            if config_name not in all_data:
                all_data[config_name] = {}

            # Append each metric across repetitions
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    if metric_name not in all_data[config_name]:
                        all_data[config_name][metric_name] = []
                    all_data[config_name][metric_name].append(metric_value)

    # Six metrics regardless of time_horizon filter
    metrics_to_analyze = [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
    ]

    # Fixed enumeration order for table rows/columns
    methods = ['Arachne_v2', 'semSegRep']
    weight_counts = [26, 52, 105]
    fitness_types = ['DISC', 'CONT', 'CONT2']

    # 6×6 groups (method + fitness), order matches reference screenshots
    groups = []
    for method in methods:
        for fitness_type in fitness_types:
            groups.append(generate_group_name(method, fitness_type))

    print(f"实际有数据的配置: {len(all_data)} 种（18种理论配置中的子集）")

    if len(all_data) == 0:
        print("\n错误: 没有提取到任何实验数据！")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("Step 2: 计算Aˆ12统计量（6×6分组表：方法+fitness；每格包含w26/w52/w105三条比较）")
    print("=" * 80)
    print("分组顺序：先方法，再fitness（例如：Arachne_v2_DISC / Arachne_v2_CONT / Arachne_v2_CONT2 / semSegRep_...）")

    # Excel output path
    if args.time_horizon:
        excel_file = f'a12_comparison_results_{args.time_horizon}.xlsx'
    else:
        excel_file = 'a12_comparison_results.xlsx'
    excel_writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    used_sheet_names = set()

    # One grouped table per metric
    for metric in metrics_to_analyze:
        print(f"\n指标: {metric}")
        print("=" * 80)

        # ----------------------------
        # Build 6×6 group table: columns are group × weight (w26/w52/w105)
        # Excel shows three sub-columns per group
        # ----------------------------
        weight_labels = [f"w{w}" for w in weight_counts]
        grouped_columns = pd.MultiIndex.from_product([groups, weight_labels])
        effect_matrix = []
        numeric_matrix = []

        for row_group in groups:
            # Method names may contain underscores (e.g. Arachne_v2); split from the right
            row_method, row_fitness = row_group.rsplit('_', 1)
            effect_row = []
            numeric_row = []

            for col_group in groups:
                col_method, col_fitness = col_group.rsplit('_', 1)

                # Diagonal: same group vs itself
                if row_group == col_group:
                    for _w in weight_counts:
                        effect_row.append("equivalent")
                        numeric_row.append(0.5)
                    continue

                for w in weight_counts:
                    row_cfg = generate_config_name(row_method, w, row_fitness)
                    col_cfg = generate_config_name(col_method, w, col_fitness)

                    if row_cfg in all_data and col_cfg in all_data and metric in all_data[row_cfg] and metric in all_data[col_cfg]:
                        v1 = all_data[row_cfg][metric]
                        v2 = all_data[col_cfg][metric]
                        if len(v1) > 0 and len(v2) > 0:
                            a12 = calculate_a12(v1, v2)
                            if a12 is None:
                                effect_row.append("N/A")
                                numeric_row.append(np.nan)
                            else:
                                effect_row.append(interpret_a12_directional(a12, row_cfg, col_cfg))
                                numeric_row.append(float(a12))
                        else:
                            effect_row.append("N/A")
                            numeric_row.append(np.nan)
                    else:
                        effect_row.append("N/A")
                        numeric_row.append(np.nan)

            effect_matrix.append(effect_row)
            numeric_matrix.append(numeric_row)

        grouped_effect_df = pd.DataFrame(effect_matrix, index=groups, columns=grouped_columns)
        grouped_numeric_df = pd.DataFrame(numeric_matrix, index=groups, columns=grouped_columns)

        # Write 6×6 sheets
        mshort = _metric_short_name(metric)
        sheet_g6_effect = _make_unique_sheet_name(f"G6_{mshort}_eff", used_sheet_names)
        sheet_g6_values = _make_unique_sheet_name(f"G6_{mshort}_val", used_sheet_names)
        grouped_effect_df.to_excel(excel_writer, sheet_name=sheet_g6_effect, index=True, merge_cells=True)
        grouped_numeric_df.to_excel(excel_writer, sheet_name=sheet_g6_values, index=True, merge_cells=True)

        # ----------------------------
        # Optional self-check
        # ----------------------------
        if args.self_check:
            # A12(X,Y) + A12(Y,X) should be 1 (ties included); diagonal 0.5
            mismatches = []
            checked = 0
            for row_group in groups:
                row_method, row_fitness = row_group.rsplit('_', 1)
                for col_group in groups:
                    col_method, col_fitness = col_group.rsplit('_', 1)
                    for w in weight_counts:
                        row_cfg = generate_config_name(row_method, w, row_fitness)
                        col_cfg = generate_config_name(col_method, w, col_fitness)

                        if row_group == col_group:
                            # Diagonal must be 0.5
                            v = _get_metric_values(all_data, row_cfg, metric)
                            if v is not None:
                                a = calculate_a12(v, v)
                                checked += 1
                                if a is None or abs(a - 0.5) > 1e-12:
                                    mismatches.append((metric, row_group, col_group, w, f"diag a12={a}"))
                            continue

                        v1 = _get_metric_values(all_data, row_cfg, metric)
                        v2 = _get_metric_values(all_data, col_cfg, metric)
                        if v1 is None or v2 is None:
                            continue  # Skip pairs without data

                        a12_12 = calculate_a12(v1, v2)
                        a12_21 = calculate_a12(v2, v1)
                        checked += 1
                        if a12_12 is None or a12_21 is None:
                            mismatches.append((metric, row_group, col_group, w, "a12 is None"))
                            continue
                        if abs((a12_12 + a12_21) - 1.0) > 1e-9:
                            mismatches.append((metric, row_group, col_group, w, f"a12+a21={a12_12+a12_21}"))

            if mismatches:
                print("\n[SELF_CHECK] 发现A12一致性异常：")
                for item in mismatches[:50]:
                    m, rg, cg, w, msg = item
                    print(f"  - metric={m}, cell={rg} vs {cg}, w{w}: {msg}")
                if len(mismatches) > 50:
                    print(f"  ... 还有 {len(mismatches)-50} 条")
                raise SystemExit("[SELF_CHECK] FAILED")
            else:
                print(f"[SELF_CHECK] PASSED for metric={metric} (checked={checked})")

    # Persist workbook
    excel_writer.close()

    # Time-horizon comparison notice
    if args.compare_horizons:
        print("\n" + "=" * 80)
        print("Step 3: Time Horizon比较（1s vs 2s vs 3s）")
        print("=" * 80)
        print("注意：由于1s和2s只有5个重复实验，而3s有10个重复实验，样本量不同")
        print("Aˆ12不适合用于不同样本量的比较，请使用calculate_cohensd.py进行time horizon比较")
        print("=" * 80)

    print("\n" + "=" * 80)
    print(f"分析完成！结果已保存到: {excel_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
