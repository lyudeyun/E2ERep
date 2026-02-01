#!/usr/bin/env python3
"""
计算 Vargha and Delaney's Aˆ12 effect size

Aˆ12 是一个非参数效应量指标，用于比较两个独立样本（非配对数据）。
Aˆ12 = P(X1 > X2) + 0.5 * P(X1 = X2)

适用场景：
- 两个独立的样本组
- 每个样本组包含多个观测值（如10次重复实验的所有值）
- 所有重复实验的值都参与计算，不需要先计算均值

取值范围：[0, 1]
- Aˆ12 = 0.5: 两个样本没有差异
- Aˆ12 > 0.5: 第一个样本倾向于大于第二个样本
- Aˆ12 < 0.5: 第一个样本倾向于小于第二个样本

效应量解释（根据Vargha & Delaney, 2000）：
- negligible: Aˆ12 ∈ (0.5, 0.556) 或 Aˆ12 ∈ (0.444, 0.5)
- small: Aˆ12 ∈ [0.556, 0.638) 或 Aˆ12 ∈ (0.362, 0.444]
- medium: Aˆ12 ∈ [0.638, 0.714) 或 Aˆ12 ∈ (0.286, 0.362]
- large: Aˆ12 ≥ 0.714 或 Aˆ12 ≤ 0.286

使用方法：
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
    计算 Vargha and Delaney's Aˆ12 effect size
    
    Parameters:
    -----------
    group1 : array-like
        第一个样本的数据
    group2 : array-like
        第二个样本的数据
    
    Returns:
    --------
    a12 : float
        Aˆ12 值，范围 [0, 1]
        - a12 = 0.5: 无差异
        - a12 > 0.5: group1 倾向于大于 group2
        - a12 < 0.5: group1 倾向于小于 group2
    """
    group1 = np.array(group1)
    group2 = np.array(group2)
    
    if len(group1) == 0 or len(group2) == 0:
        return None
    
    n1 = len(group1)
    n2 = len(group2)
    
    # 方法1：使用秩和计算（更高效）
    # 合并两个样本并计算秩
    combined = np.concatenate([group1, group2])
    ranks = stats.rankdata(combined, method='average')  # 使用平均秩处理相同值
    
    # 计算第一个样本的秩和
    r1 = np.sum(ranks[:n1])
    
    # Aˆ12 = (R1 - n1(n1+1)/2) / (n1 * n2)
    a12 = (r1 - n1 * (n1 + 1) / 2) / (n1 * n2)
    
    return a12

def _get_metric_values(all_data, config_name, metric):
    """安全获取某个配置在某个metric上的所有重复实验值（list），拿不到就返回None"""
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
    解释 Aˆ12 的效应量大小
    
    根据 Vargha & Delaney (2000) 的阈值：
    - negligible: Aˆ12 ∈ (0.5, 0.556) 或 Aˆ12 ∈ (0.444, 0.5)
    - small: Aˆ12 ∈ [0.556, 0.638) 或 Aˆ12 ∈ (0.362, 0.444]
    - medium: Aˆ12 ∈ [0.638, 0.714) 或 Aˆ12 ∈ (0.286, 0.362]
    - large: Aˆ12 ≥ 0.714 或 Aˆ12 ≤ 0.286
    
    Parameters:
    -----------
    a12 : float
        Aˆ12 值
    
    Returns:
    --------
    interpretation : str
        效应量解释
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
    解释 Aˆ12 的效应量大小（带方向）
    
    在18×18表格中：
    - 行是row_config（行配置）
    - 列是col_config（列配置）
    - 所有metric都是越小越好
    - Aˆ12 > 0.5: 行配置的值倾向于更大 → 行配置更差（返回xxx_worse）
    - Aˆ12 < 0.5: 行配置的值倾向于更小 → 行配置更好（返回xxx_better）
    
    根据 Vargha & Delaney (2000) 的阈值：
    - negligible: Aˆ12 ∈ (0.5, 0.556) 或 Aˆ12 ∈ (0.444, 0.5)
    - small: Aˆ12 ∈ [0.556, 0.638) 或 Aˆ12 ∈ (0.362, 0.444]
    - medium: Aˆ12 ∈ [0.638, 0.714) 或 Aˆ12 ∈ (0.286, 0.362]
    - large: Aˆ12 ≥ 0.714 或 Aˆ12 ≤ 0.286
    
    Parameters:
    -----------
    a12 : float
        Aˆ12 值
    row_config : str, optional
        行配置名称（用于生成更清晰的描述）
    col_config : str, optional
        列配置名称（用于生成更清晰的描述）
    
    Returns:
    --------
    interpretation : str
        效应量解释（带方向）
    """
    if a12 is None:
        return "N/A"
    
    if a12 == 0.5:
        return "equivalent"
    elif a12 > 0.5:
        # Aˆ12 > 0.5 意味着行配置的值倾向于大于列配置
        # 对于"越小越好"的metric，行配置更差，列配置更好
        if a12 < 0.556:
            return "negligible_worse"
        elif a12 < 0.638:
            return "small_worse"
        elif a12 < 0.714:
            return "medium_worse"
        else:  # a12 >= 0.714
            return "large_worse"
    else:  # a12 < 0.5
        # Aˆ12 < 0.5 意味着行配置的值倾向于小于列配置
        # 对于"越小越好"的metric，行配置更好，列配置更差
        if a12 > 0.444:
            return "negligible_better"
        elif a12 > 0.362:
            return "small_better"
        elif a12 > 0.286:
            return "medium_better"
        else:  # a12 <= 0.286
            return "large_better"


def extract_metrics_from_log(log_file):
    """从log文件中提取指标"""
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 提取L2误差
            for horizon in [1, 2, 3]:
                pattern = rf'plan_L2_{horizon}s:([\d.e+-]+)'
                match = re.search(pattern, content)
                if match:
                    metrics[f'plan_L2_{horizon}s'] = float(match.group(1))
                else:
                    metrics[f'plan_L2_{horizon}s'] = None
            
            # 提取碰撞率
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
    解析实验文件夹名称
    格式：VAD_base_REP_VAL_{method}_DE_w{weight_count}_p*_i*_es*_{fitness_type}_{repetition}
    """
    method = None
    weight_count = None
    fitness_type = None
    repetition = None
    
    # 提取方法
    if 'Arachne_v2' in folder_name:
        method = 'Arachne_v2'
    elif 'semSegRep' in folder_name:
        method = 'semSegRep'
    
    # 提取权重数量
    weight_match = re.search(r'_w(\d+)_', folder_name)
    if weight_match:
        weight_count = int(weight_match.group(1))
    
    # 提取fitness类型
    if '_DISC_' in folder_name:
        fitness_type = 'DISC'
    elif '_CONT2_' in folder_name:
        fitness_type = 'CONT2'
    elif '_CONT_' in folder_name:
        fitness_type = 'CONT'
    
    # 提取重复次数
    rep_match = re.search(r'_(CONT|CONT2|DISC)_(\d+)$', folder_name)
    if rep_match:
        repetition = int(rep_match.group(2))
    
    return method, weight_count, fitness_type, repetition


def generate_config_name(method, weight_count, fitness_type):
    """生成配置名称"""
    return f"{method}_w{weight_count}_{fitness_type}"

def generate_group_name(method, fitness_type):
    """生成分组名称（方法 + fitness），用于6×6大表"""
    return f"{method}_{fitness_type}"

def _metric_short_name(metric: str) -> str:
    """
    将metric压缩成更短的名字，避免Excel sheet name超过31字符限制
    """
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
    Excel sheet name长度限制为31。这里做：
    - 截断到<=31
    - 若重名，追加 _1/_2...
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
    """计算所有18种配置之间的Aˆ12比较"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='计算Aˆ12统计量')
    parser.add_argument('--time_horizon', type=str, choices=['1s', '2s', '3s'], default=None,
                        help='指定time horizon，只读取该目录下的数据（默认：读取所有）')
    parser.add_argument('--compare_horizons', action='store_true',
                        help='进行time horizon之间的比较（1s vs 2s vs 3s）')
    parser.add_argument('--self_check', action='store_true',
                        help='对6×6表格内部的A12计算做一致性自检（反向比较之和应为1，对角线应为0.5）')
    args = parser.parse_args()
    
    # 实验根目录
    base_dirs = [
        '/home/deyun/git/B2DRepair/vad_base_Arachne_v2_DE_results',
        '/home/deyun/git/B2DRepair/vad_base_semSegRep_DE_results'
    ]
    
    # 存储所有数据：config -> metric -> [所有重复实验的值]
    all_data = {}
    # 存储time horizon数据：config -> repetition -> time_horizon -> metric -> value
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
        
        # 如果指定了time_horizon，只在该目录下查找
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
        # 递归查找所有以 VAD_base_REP_VAL_ 开头的文件夹
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
                continue
            
            # 提取指标
            metrics = extract_metrics_from_log(log_file)
            
            if metrics is None:
                continue
            
            # 生成配置名称
            config_name = generate_config_name(method, weight_count, fitness_type)
            
            # 存储time horizon数据（用于time horizon比较）
            if time_horizon:
                if config_name not in time_horizon_data:
                    time_horizon_data[config_name] = {}
                if repetition not in time_horizon_data[config_name]:
                    time_horizon_data[config_name][repetition] = {}
                time_horizon_data[config_name][repetition][time_horizon] = metrics
            
            # 初始化配置数据
            if config_name not in all_data:
                all_data[config_name] = {}
            
            # 存储每个metric的值（所有重复实验）
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    if metric_name not in all_data[config_name]:
                        all_data[config_name][metric_name] = []
                    all_data[config_name][metric_name].append(metric_value)
    
    # 定义要分析的指标（总是6个metric，不管time_horizon是什么）
    metrics_to_analyze = [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
    ]
    
    # 固定的枚举顺序（用于输出表格的行/列顺序）
    methods = ['Arachne_v2', 'semSegRep']
    weight_counts = [26, 52, 105]
    fitness_types = ['DISC', 'CONT', 'CONT2']
    
    # 6×6分组（方法 + fitness），顺序与截图一致
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
    
    # 创建Excel文件（根据time_horizon调整文件名）
    if args.time_horizon:
        excel_file = f'a12_comparison_results_{args.time_horizon}.xlsx'
    else:
    excel_file = 'a12_comparison_results.xlsx'
    excel_writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    used_sheet_names = set()
    
    # 为每个metric生成18×18表格
    for metric in metrics_to_analyze:
        print(f"\n指标: {metric}")
        print("=" * 80)
        
        # ----------------------------
        # 先生成 6×6 分组表（方法+fitness），列按“分组 × 权重子列(w26/w52/w105)”展开
        # 这样Excel里会显示为：每个分组列下面有3个子列（w26/w52/w105），符合截图格式。
        # ----------------------------
        weight_labels = [f"w{w}" for w in weight_counts]
        grouped_columns = pd.MultiIndex.from_product([groups, weight_labels])
        effect_matrix = []
        numeric_matrix = []

        for row_group in groups:
            # 注意：method 里可能包含下划线（例如 Arachne_v2），所以必须从右侧切分
            row_method, row_fitness = row_group.rsplit('_', 1)
            effect_row = []
            numeric_row = []

            for col_group in groups:
                col_method, col_fitness = col_group.rsplit('_', 1)

                # 对角线：同组 vs 同组
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

        # 写入 Excel（6×6）
        mshort = _metric_short_name(metric)
        sheet_g6_effect = _make_unique_sheet_name(f"G6_{mshort}_eff", used_sheet_names)
        sheet_g6_values = _make_unique_sheet_name(f"G6_{mshort}_val", used_sheet_names)
        grouped_effect_df.to_excel(excel_writer, sheet_name=sheet_g6_effect, index=True, merge_cells=True)
        grouped_numeric_df.to_excel(excel_writer, sheet_name=sheet_g6_values, index=True, merge_cells=True)
        
        # ----------------------------
        # 可选：一致性自检
        # ----------------------------
        if args.self_check:
            # A12(X,Y) + A12(Y,X) 应为 1（含tie的0.5项也成立），对角线应为0.5
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
                            # 对角线：必须等价（0.5）
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
                            continue  # 没数据的不检

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
    
    # 保存Excel文件
    excel_writer.close()
    
    # Time Horizon比较
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

