#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-run open-loop evaluation for experiments where repair succeeded but eval failed.

Usage:
    # Batch: --exp-dir is a parent folder containing many experiment subdirs
    python3 batch_reeval_openloop.py \
        --exp-dir /data1/uniad_tiny_Arachne_v2_DE_results \
        --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
        --eval-cuda-device 0 \
        [--jobs 3] \
        [--occ-output-dir baseline/UniAD/uniad_occ_cache] \
        [--dry-run]

    # Single experiment: --exp-dir points to one folder with repair/repair_output
    python3 batch_reeval_openloop.py \
        --exp-dir /data1/uniad_tiny_semSegRep_DE_results/UniAD_tiny_REP_VAL_t3s_semSegRep_DE_w7_p14_i50_es5_DISC_1 \
        --eval-dataset Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl \
        --eval-cuda-device 0
"""

import sys
import argparse
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import functions from run_experiment.py
sys.path.insert(0, str(Path(__file__).parent))
from run_experiment import run_evaluation


def parse_exp_dir_name(exp_dir_name):
    """
    Parse metadata from an experiment directory name.

    Pattern: {model_name}_REP_VAL_{time_horizon_str}_{rep_method}_{search_algo}_{search_params}_{fitness_str}_{run_idx}
    Example: UniAD_tiny_REP_VAL_t3s_Arachne_v2_DE_w13_p26_i50_es5_DISC_1

    Returns:
        dict with keys: model_name, model_type, or None if cannot parse
    """
    # Match model_name (VAD_base, UniAD_tiny, etc.)
    match = re.match(r'^(VAD_|UniAD_)(base|tiny)', exp_dir_name)
    if not match:
        return None
    
    model_name = match.group(0)  # e.g., "UniAD_tiny"
    model_type = "VAD" if model_name.startswith("VAD_") else "UniAD"
    
    return {
        'model_name': model_name,
        'model_type': model_type,
    }


def find_repaired_model(exp_dir, model_type):
    """Return path to repaired weights under repair/repair_output/."""
    repair_dir = exp_dir / "repair" / "repair_output"
    
    if model_type == "VAD":
        model_file = repair_dir / "VAD_repaired_both_layers.pth"
    elif model_type == "UniAD":
        model_file = repair_dir / "repaired_model.pth"
    else:
        return None
    
    if model_file.exists():
        return model_file.resolve()
    return None


def check_eval_complete(exp_dir, model_name):
    """True if open-loop eval artifacts (.json and .log) already exist."""
    eval_dir = exp_dir / "open_loop_eval"
    model_lower = model_name.lower()
    
    json_file = eval_dir / f"{model_lower}_rep_val.json"
    log_file = eval_dir / f"{model_lower}_rep_val.log"
    
    return json_file.exists() and log_file.exists()


def main():
    parser = argparse.ArgumentParser(
        description="Batch re-run open-loop evaluation"
    )
    parser.add_argument(
        '--exp-dir',
        type=str,
        required=True,
        help='Experiment root (batch) or single experiment dir with repair/repair_output'
    )
    parser.add_argument(
        '--eval-dataset',
        type=str,
        required=True,
        help='Eval dataset PKL (e.g. Bench2DriveZoo/data/infos/b2d_infos_val_partB_25clips.pkl)'
    )
    parser.add_argument(
        '--eval-cuda-device',
        type=str,
        default='0',
        help='CUDA device id (default: 0)'
    )
    parser.add_argument(
        '--occ-output-dir',
        type=str,
        default=None,
        help='Optional occupancy cache dir for collision scoring during eval'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Parallel eval workers on one GPU (default 1 = serial)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print which experiments would run without executing'
    )

    args = parser.parse_args()
    
    repo_root = Path(__file__).parent.absolute()
    exp_base_dir = Path(args.exp_dir).resolve()
    
    if not exp_base_dir.exists():
        print(f"ERROR: 实验目录不存在: {exp_base_dir}")
        return 1
    
    eval_dataset = Path(args.eval_dataset)
    if not eval_dataset.is_absolute():
        eval_dataset = repo_root / eval_dataset
    eval_dataset = eval_dataset.resolve()
    
    if not eval_dataset.exists():
        print(f"ERROR: 评估数据集不存在: {eval_dataset}")
        return 1
    
    # Scan experiments: if path itself is one experiment, only that; else scan children
    print(f"扫描实验目录: {exp_base_dir}")
    print("=" * 80)
    
    exp_dirs_to_reeval = []
    info_self = parse_exp_dir_name(exp_base_dir.name)
    if info_self is not None and (exp_base_dir / "repair" / "repair_output").is_dir():
        # --exp-dir is a single experiment directory
        candidates = [exp_base_dir]
        print("  (检测到单个实验目录，仅评估该目录)")
    else:
        # --exp-dir is a parent; scan immediate subdirectories
        candidates = [item for item in exp_base_dir.iterdir() if item.is_dir()]
    
    for item in candidates:
        exp_dir_name = item.name
        
        # Parse folder metadata
        info = parse_exp_dir_name(exp_dir_name)
        if info is None:
            print(f"  [跳过] {exp_dir_name} (无法解析目录名)")
            continue
        
        model_name = info['model_name']
        model_type = info['model_type']
        
        # Require repaired checkpoint
        repaired_model = find_repaired_model(item, model_type)
        if repaired_model is None:
            print(f"  [跳过] {exp_dir_name} (没有修复后的模型)")
            continue
        
        # Skip if eval already finished
        if check_eval_complete(item, model_name):
            print(f"  [跳过] {exp_dir_name} (评估已完成)")
            continue
        
        # Needs re-eval
        print(f"  [需要重跑] {exp_dir_name}")
        exp_dirs_to_reeval.append({
            'exp_dir': item,
            'model_name': model_name,
            'model_type': model_type,
            'repaired_model': repaired_model,
        })
    
    print("=" * 80)
    print(f"找到 {len(exp_dirs_to_reeval)} 个需要重跑的实验")
    
    if len(exp_dirs_to_reeval) == 0:
        print("没有需要重跑的实验，退出。")
        return 0
    
    if args.dry_run:
        print("\n[DRY RUN] 以下实验将被重跑:")
        for i, exp_info in enumerate(exp_dirs_to_reeval, 1):
            print(f"  {i}. {exp_info['exp_dir'].name}")
        return 0
    
    # Fan out evaluations (optional parallelism on one GPU)
    njobs = max(1, int(args.jobs))
    print(f"\n开始批量重跑 (jobs={njobs}, 单 GPU: {args.eval_cuda_device})...")
    print("=" * 80)

    def run_one(exp_info):
        exp_dir = exp_info['exp_dir']
        model_name = exp_info['model_name']
        model_type = exp_info['model_type']
        repaired_model = exp_info['repaired_model']
        try:
            success = run_evaluation(
                model_name=model_name,
                model_type=model_type,
                repaired_model=repaired_model,
                eval_dataset=eval_dataset,
                exp_dir=exp_dir,
                repo_root=repo_root,
                cuda_device=args.eval_cuda_device,
                use_tmpfs_data=False,
                occ_output_dir=args.occ_output_dir,
            )
            return (exp_dir.name, success, None)
        except Exception as e:
            import traceback
            return (exp_dir.name, False, traceback.format_exc())

    success_count = 0
    fail_count = 0

    if njobs <= 1:
        for i, exp_info in enumerate(exp_dirs_to_reeval, 1):
            exp_dir = exp_info['exp_dir']
            print(f"\n[{i}/{len(exp_dirs_to_reeval)}] 重跑: {exp_dir.name}")
            print("-" * 80)
            name, ok, err = run_one(exp_info)
            if ok:
                success_count += 1
                print(f"✅ {name} 评估成功")
            else:
                fail_count += 1
                print(f"❌ {name} 评估失败")
                if err:
                    print(err)
    else:
        with ThreadPoolExecutor(max_workers=njobs) as executor:
            fut_to_info = {executor.submit(run_one, exp_info): exp_info for exp_info in exp_dirs_to_reeval}
            done = 0
            for fut in as_completed(fut_to_info):
                done += 1
                name, ok, err = fut.result()
                if ok:
                    success_count += 1
                    print(f"[{done}/{len(exp_dirs_to_reeval)}] ✅ {name} 评估成功")
                else:
                    fail_count += 1
                    print(f"[{done}/{len(exp_dirs_to_reeval)}] ❌ {name} 评估失败")
                    if err:
                        print(err)

    # Summary
    print("\n" + "=" * 80)
    print(f"批量重跑完成:")
    print(f"  成功: {success_count}/{len(exp_dirs_to_reeval)}")
    print(f"  失败: {fail_count}/{len(exp_dirs_to_reeval)}")
    print("=" * 80)
    
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
