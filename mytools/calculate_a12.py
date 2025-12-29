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


def main():
    """计算所有18种配置之间的Aˆ12比较"""
    # 实验根目录
    base_dirs = [
        '/home/deyun/git/B2DRepair/vad_base_Arachne_v2_DE_results',
        '/home/deyun/git/B2DRepair/vad_base_semSegRep_DE_results'
    ]
    
    # 存储所有数据：config -> metric -> [所有重复实验的值]
    all_data = {}
    
    print("=" * 80)
    print("Step 1: 提取所有实验的指标数据")
    print("=" * 80)
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            continue
        
        print(f"\n处理目录: {base_dir}")
        base_path = Path(base_dir)
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
            
            # 初始化配置数据
            if config_name not in all_data:
                all_data[config_name] = {}
            
            # 存储每个metric的值（所有重复实验）
            for metric_name, metric_value in metrics.items():
                if metric_value is not None:
                    if metric_name not in all_data[config_name]:
                        all_data[config_name][metric_name] = []
                    all_data[config_name][metric_name].append(metric_value)
    
    # 定义要分析的指标
    metrics_to_analyze = [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
    ]
    
    # 生成所有配置的列表（按固定顺序：方法 -> 权重 -> fitness）
    methods = ['Arachne_v2', 'semSegRep']
    weight_counts = [26, 52, 105]
    fitness_types = ['DISC', 'CONT', 'CONT2']
    
    configs = []
    for method in methods:
        for weight_count in weight_counts:
            for fitness_type in fitness_types:
                config_name = generate_config_name(method, weight_count, fitness_type)
                configs.append(config_name)
    
    print(f"\n总共找到 {len(configs)} 种配置")
    print(f"实际有数据的配置: {len(all_data)} 种")
    
    # 检查每个配置是否有数据
    missing_configs = []
    for config in configs:
        if config not in all_data:
            missing_configs.append(config)
        else:
            for metric in metrics_to_analyze:
                if metric not in all_data[config] or len(all_data[config][metric]) == 0:
                    missing_configs.append(f"{config}_{metric}")
                    break
    
    if missing_configs:
        print(f"\n警告: 以下配置缺少数据:")
        for config in missing_configs[:20]:
            print(f"  - {config}")
        if len(missing_configs) > 20:
            print(f"  ... 还有 {len(missing_configs) - 20} 个")
    
    if len(all_data) == 0:
        print("\n错误: 没有提取到任何实验数据！")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Step 2: 计算Aˆ12统计量（18×18比较表格）")
    print("=" * 80)
    print("18种配置 = 2个方法 × 3个权重 × 3个fitness函数")
    print("配置顺序：先方法，再权重，再fitness")
    
    # 创建Excel文件
    excel_file = 'a12_comparison_results.xlsx'
    excel_writer = pd.ExcelWriter(excel_file, engine='openpyxl')
    
    # 为每个metric生成18×18表格
    for metric in metrics_to_analyze:
        print(f"\n指标: {metric}")
        print("=" * 80)
        
        # 构建18×18矩阵
        matrix = np.full((18, 18), np.nan)
        effect_size_matrix = {}
        
        for i, config1 in enumerate(configs):
            for j, config2 in enumerate(configs):
                if i == j:
                    # 对角线：自己和自己比较
                    matrix[i, j] = 0.5
                    effect_size_matrix[(i, j)] = "equivalent"
                else:
                    # 获取两个配置的所有重复实验值
                    if config1 in all_data and config2 in all_data:
                        if metric in all_data[config1] and metric in all_data[config2]:
                            values1 = all_data[config1][metric]
                            values2 = all_data[config2][metric]
                            
                            if len(values1) > 0 and len(values2) > 0:
                                a12 = calculate_a12(values1, values2)
                                if a12 is not None:
                                    matrix[i, j] = a12
                                    effect_size = interpret_a12_directional(a12, config1, config2)
                                    effect_size_matrix[(i, j)] = effect_size
        
        # 打印表格（显示效应量解释）
        print("\n18×18 Aˆ12 比较表格（显示效应量解释）:")
        print("说明：行配置 vs 列配置（所有metric都是越小越好）")
        print("  - better: 行配置更好（行配置的值更小，Aˆ12 < 0.5）")
        print("  - worse: 行配置更差（行配置的值更大，Aˆ12 > 0.5）")
        print("  - equivalent: 无差异（Aˆ12 = 0.5）")
        print()
        
        # 计算最大配置名称长度
        max_config_len = max([len(config) for config in configs]) if configs else 20
        max_config_len = max(max_config_len, 20)  # 至少20个字符
        
        # 表头
        header = f"{'':{max_config_len}s}"
        for config in configs:
            header += f"\t{config:{max_config_len}s}"
        print(header)
        print("-" * len(header))
        
        for i, config1 in enumerate(configs):
            row_str = f"{config1:{max_config_len}s}"
            for j, config2 in enumerate(configs):
                if (i, j) in effect_size_matrix:
                    row_str += f"\t{effect_size_matrix[(i, j)]:{max_config_len}s}"
                elif not np.isnan(matrix[i, j]):
                    row_str += f"\t{matrix[i, j]:.4f}"
                else:
                    row_str += f"\t{'N/A':{max_config_len}s}"
            print(row_str)
        
        # 也可以打印数值表格
        print("\n18×18 Aˆ12 比较表格（显示数值）:")
        print("说明：Aˆ12 = P(行配置 > 列配置) + 0.5 * P(行配置 = 列配置)")
        print("  - Aˆ12 > 0.5: 行配置的值倾向于更大 → 行配置更差（metric越小越好）")
        print("  - Aˆ12 < 0.5: 行配置的值倾向于更小 → 行配置更好（metric越小越好）")
        print("  - Aˆ12 = 0.5: 无差异")
        print()
        
        # 使用相同的最大配置名称长度
        header = f"{'':{max_config_len}s}"
        for config in configs:
            header += f"\t{config:{max_config_len}s}"
        print(header)
        print("-" * len(header))
        
        for i, config1 in enumerate(configs):
            row_str = f"{config1:{max_config_len}s}"
            for j, config2 in enumerate(configs):
                if not np.isnan(matrix[i, j]):
                    row_str += f"\t{matrix[i, j]:.4f}"
                else:
                    row_str += f"\t{'N/A':{max_config_len}s}"
            print(row_str)
        
        # 创建DataFrame并写入Excel
        # 效应量解释表格
        effect_size_data = []
        for i, config1 in enumerate(configs):
            row_data = [config1]
            for j, config2 in enumerate(configs):
                if (i, j) in effect_size_matrix:
                    row_data.append(effect_size_matrix[(i, j)])
                elif not np.isnan(matrix[i, j]):
                    row_data.append(f"{matrix[i, j]:.4f}")
                else:
                    row_data.append("N/A")
            effect_size_data.append(row_data)
        
        effect_size_df = pd.DataFrame(effect_size_data, columns=[''] + configs)
        sheet_name_effect = f"{metric}_effect_size"
        effect_size_df.to_excel(excel_writer, sheet_name=sheet_name_effect, index=False)
        
        # 数值表格
        numeric_data = []
        for i, config1 in enumerate(configs):
            row_data = [config1]
            for j, config2 in enumerate(configs):
                if not np.isnan(matrix[i, j]):
                    row_data.append(matrix[i, j])
                else:
                    row_data.append(np.nan)
            numeric_data.append(row_data)
        
        numeric_df = pd.DataFrame(numeric_data, columns=[''] + configs)
        sheet_name_numeric = f"{metric}_values"
        numeric_df.to_excel(excel_writer, sheet_name=sheet_name_numeric, index=False)
    
    # 保存Excel文件
    excel_writer.close()
    
    print("\n" + "=" * 80)
    print(f"分析完成！结果已保存到: {excel_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()

