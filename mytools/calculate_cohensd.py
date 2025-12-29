#!/usr/bin/env python3
"""
计算Cohen's d统计量，用于比较不同实验配置的效果

使用方法：
conda activate b2d_zoo
python calculate_cohensd.py

实验结构：
- 两个方法：Arachne_v2, semSegRep
- 三个fitness函数：DISC, CONT, CONT2
- 每个实验重复5次
- 三种权重选择：w26 (26个权重), w52 (52个权重), w105 (105个权重)

提取的指标：
- plan_L2_1s, plan_L2_2s, plan_L2_3s (L2误差)
- plan_obj_box_col_1s, plan_obj_box_col_2s, plan_obj_box_col_3s (碰撞率)
"""

import os
import re
import json
import sys

try:
    import numpy as np
    import pandas as pd
except ImportError:
    print("Error: numpy and pandas are required. Please activate b2d_zoo conda environment:")
    print("  conda activate b2d_zoo")
    sys.exit(1)

from pathlib import Path
from collections import defaultdict

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
    # 实验根目录
    base_dirs = [
        '/home/deyun/git/B2DRepair/vad_base_Arachne_v2_DE_results',
        '/home/deyun/git/B2DRepair/vad_base_semSegRep_DE_results'
    ]
    
    # 存储所有数据
    all_data = []
    missing_log = []  # 缺少log文件的实验
    parse_failed = []  # 无法解析配置的实验
    extract_failed = []  # 无法提取指标的实验
    success_count = 0  # 成功提取的实验数
    
    print("=" * 80)
    print("Step 1: 提取所有实验的指标数据")
    print("=" * 80)
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            continue
        
        print(f"\n处理目录: {base_dir}")
        # 递归查找所有子目录中的实验文件夹（支持 large/middle/small 子目录结构）
        exp_folders = []
        base_path = Path(base_dir)
        # 递归查找所有以 VAD_base_REP_VAL_ 开头的文件夹
        for exp_folder in base_path.rglob('VAD_base_REP_VAL_*'):
            if exp_folder.is_dir():
                exp_folders.append(exp_folder)
        
        exp_folders = sorted(exp_folders)
        print(f"  找到 {len(exp_folders)} 个实验文件夹")
        
        for exp_folder in exp_folders:
            log_file = exp_folder / 'open_loop_eval' / 'vad_base_rep_val.log'
            
            if not log_file.exists():
                missing_log.append(str(exp_folder.relative_to(base_path)))
                continue
            
            # 解析实验配置
            method, weight_count, fitness_type, repetition = parse_experiment_name(exp_folder.name)
            
            if not all([method, weight_count, fitness_type, repetition]):
                parse_failed.append(str(exp_folder.relative_to(base_path)))
                continue
            
            # 提取指标
            metrics = extract_metrics_from_log(log_file)
            
            if metrics is None:
                extract_failed.append(str(exp_folder.relative_to(base_path)))
                continue
            
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
        print("2. 实验文件夹中缺少 open_loop_eval/vad_base_rep_val.log 文件")
        print("3. log文件中不包含所需的指标数据（plan_L2_*s 或 plan_obj_box_col_*s）")
        print("4. 实验文件夹名称格式不正确，无法解析配置")
        print("\n脚本已停止执行。")
        sys.exit(1)
    
    # 转换为DataFrame便于分析
    df = pd.DataFrame(all_data)
    
    # 定义要分析的指标
    metrics_to_analyze = [
        'plan_L2_1s', 'plan_L2_2s', 'plan_L2_3s',
        'plan_obj_box_col_1s', 'plan_obj_box_col_2s', 'plan_obj_box_col_3s'
    ]
    
    print("\n" + "=" * 80)
    print("Step 2: 计算Cohen's d统计量")
    print("=" * 80)
    
    results = []
    
    # 比较1: 不同方法之间的比较 (Arachne_v2 vs semSegRep)
    # 使用配对比较：先计算每种配置的均值，然后配对比较
    print("\n--- 比较1: 方法比较 (Arachne_v2 vs semSegRep) ---")
    print("  9×1向量维度说明：每个维度代表一个配置 (weight_count, fitness_type)")
    print("  维度顺序：(w26, DISC), (w26, CONT), (w26, CONT2), (w52, DISC), (w52, CONT), (w52, CONT2), (w105, DISC), (w105, CONT), (w105, CONT2)")
    for metric in metrics_to_analyze:
        # 收集所有配置的均值
        arachne_means = []
        semsegrep_means = []
        configs = []  # 记录配置信息用于显示
        
        for weight_count in [26, 52, 105]:
            for fitness_type in ['DISC', 'CONT', 'CONT2']:
                # 计算Arachne_v2的均值
                arachne_values = df[(df['method'] == 'Arachne_v2') & 
                                   (df['fitness_type'] == fitness_type) & 
                                   (df['weight_count'] == weight_count)][metric].dropna().values
                # 计算semSegRep的均值
                semsegrep_values = df[(df['method'] == 'semSegRep') & 
                                     (df['fitness_type'] == fitness_type) & 
                                     (df['weight_count'] == weight_count)][metric].dropna().values
                
                if len(arachne_values) > 0 and len(semsegrep_values) > 0:
                    arachne_mean = np.mean(arachne_values)
                    semsegrep_mean = np.mean(semsegrep_values)
                    arachne_means.append(arachne_mean)
                    semsegrep_means.append(semsegrep_mean)
                    configs.append((weight_count, fitness_type))
        
        # 如果有配对数据，计算配对样本的Cohen's d
        if len(arachne_means) > 0 and len(semsegrep_means) == len(arachne_means):
            arachne_array = np.array(arachne_means)  # 9×1向量：9个配置的均值
            semsegrep_array = np.array(semsegrep_means)  # 9×1向量：9个配置的均值
            d = cohens_d(arachne_array, semsegrep_array, paired=True)
            
            # 计算两个向量的均值和标准差（用于显示）
            mean1, mean2 = np.mean(arachne_array), np.mean(semsegrep_array)
            std1 = np.std(arachne_array, ddof=1) if len(arachne_array) > 1 else 0
            std2 = np.std(semsegrep_array, ddof=1) if len(semsegrep_array) > 1 else 0
            
            results.append({
                'comparison_type': 'Method',
                'group1': 'Arachne_v2',
                'group2': 'semSegRep',
                'fitness_type': None,  # 这是所有配置的汇总
                'weight_count': None,
                'metric': metric,
                'cohens_d': d,
                'mean1': mean1,
                'mean2': mean2,
                'std1': std1,
                'std2': std2,
                'n1': len(arachne_array),
                'n2': len(semsegrep_array)
            })
            
            if d is not None:
                print(f"\n  {metric}:")
                print(f"    Arachne_v2向量 (9×1): {arachne_array}")
                print(f"    semSegRep向量 (9×1): {semsegrep_array}")
    
    # 比较2: 不同fitness函数之间的比较
    print("\n--- 比较2: Fitness函数比较 ---")
    print("  6×1向量维度说明：每个维度代表一个配置 (method, weight_count)")
    print("  维度顺序：(Arachne_v2, w26), (Arachne_v2, w52), (Arachne_v2, w105), (semSegRep, w26), (semSegRep, w52), (semSegRep, w105)")
    fitness_types = ['DISC', 'CONT', 'CONT2']
    
    for metric in metrics_to_analyze:
        # 为每个fitness收集6×1向量（2个method × 3个weight_count）
        fitness_vectors = {}
        for fitness_type in fitness_types:
            means = []
            for method in ['Arachne_v2', 'semSegRep']:
                for weight_count in [26, 52, 105]:
                    values = df[(df['method'] == method) & 
                               (df['fitness_type'] == fitness_type) & 
                               (df['weight_count'] == weight_count)][metric].dropna().values
                    if len(values) > 0:
                        means.append(np.mean(values))
            if len(means) == 6:  # 确保有完整的6个值
                fitness_vectors[fitness_type] = np.array(means)
        
        # 显示3个fitness各自的6×1向量
        if len(fitness_vectors) == 3:
            print(f"\n  {metric}:")
            for fitness_type in fitness_types:
                if fitness_type in fitness_vectors:
                    print(f"    {fitness_type}向量 (6×1): {fitness_vectors[fitness_type]}")
            
            # 计算配对Cohen's d并保存到results（不显示）
            for i, fitness1 in enumerate(fitness_types):
                for j, fitness2 in enumerate(fitness_types):
                    if i != j and fitness1 in fitness_vectors and fitness2 in fitness_vectors:
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
                                'mean1': np.mean(vec1),
                                'mean2': np.mean(vec2),
                                'std1': np.std(vec1, ddof=1),
                                'std2': np.std(vec2, ddof=1),
                                'n1': len(vec1),
                                'n2': len(vec2)
                            })
    
    # 比较3: 不同权重数量之间的比较
    print("\n--- 比较3: 权重数量比较 ---")
    print("  6×1向量维度说明：每个维度代表一个配置 (method, fitness_type)")
    print("  维度顺序：(Arachne_v2, DISC), (Arachne_v2, CONT), (Arachne_v2, CONT2), (semSegRep, DISC), (semSegRep, CONT), (semSegRep, CONT2)")
    weight_counts = [26, 52, 105]
    weight_names = ['w26', 'w52', 'w105']
    
    for metric in metrics_to_analyze:
        # 为每个权重收集6×1向量（2个method × 3个fitness_type）
        weight_vectors = {}
        for weight_count, weight_name in zip(weight_counts, weight_names):
            means = []
            for method in ['Arachne_v2', 'semSegRep']:
                for fitness_type in ['DISC', 'CONT', 'CONT2']:
                    values = df[(df['method'] == method) & 
                               (df['fitness_type'] == fitness_type) & 
                               (df['weight_count'] == weight_count)][metric].dropna().values
                    if len(values) > 0:
                        means.append(np.mean(values))
            if len(means) == 6:  # 确保有完整的6个值
                weight_vectors[weight_name] = np.array(means)
        
        # 显示3个权重各自的6×1向量
        if len(weight_vectors) == 3:
            print(f"\n  {metric}:")
            for weight_name in weight_names:
                if weight_name in weight_vectors:
                    print(f"    {weight_name}向量 (6×1): {weight_vectors[weight_name]}")
            
            # 计算配对Cohen's d并保存到results（不显示）
            for i, weight1 in enumerate(weight_names):
                for j, weight2 in enumerate(weight_names):
                    if i != j and weight1 in weight_vectors and weight2 in weight_vectors:
                        vec1 = weight_vectors[weight1]
                        vec2 = weight_vectors[weight2]
                        d = cohens_d(vec1, vec2, paired=True)
                        if d is not None:
                            results.append({
                                'comparison_type': 'Weight',
                                'group1': weight1,
                                'group2': weight2,
                                'method': None,
                                'fitness_type': None,
                                'metric': metric,
                                'cohens_d': d,
                                'mean1': np.mean(vec1),
                                'mean2': np.mean(vec2),
                                'std1': np.std(vec1, ddof=1),
                                'std2': np.std(vec2, ddof=1),
                                'n1': len(vec1),
                                'n2': len(vec2)
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
                                    row_str += "N/A".center(16)
                        print(row_str)
                elif comparison_type == 'Weight':
                    # Weight比较：显示3×3矩阵
                    weight_names = ['w26', 'w52', 'w105']
                    # 构建矩阵
                    matrix = {}
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            effect_size = interpret_cohens_d(d)
                            key = (row['group1'], row['group2'])
                            matrix[key] = effect_size
                    
                    print(f"指标：{metric}；w26 vs w52 vs w105")
                    print("    " + " " * 12 + "w26" + " " * 12 + "w52" + " " * 12 + "w105")
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
                                    row_str += "N/A".center(16)
                        print(row_str)
                else:
                    # Method比较：一行显示
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            # 使用新规则解释效应量大小
                            effect_size = interpret_cohens_d(d)
                            
                            print(f"指标：{metric}；{row['group1']} vs {row['group2']}：{effect_size}")
    
    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()

