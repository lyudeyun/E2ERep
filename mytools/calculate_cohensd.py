#!/usr/bin/env python3
"""
Compute Cohen's d to compare experiment configurations.

Usage:
conda activate b2d_zoo
python calculate_cohensd.py [--model-type vad|uniad]

- VAD: read plan_L2_* / plan_obj_box_col_* from open_loop_eval/<model>_rep_val.log.
- UniAD: convert open_loop_eval/<model>_rep_val.json with baseline/convert_uniad_to_vad_metrics.py
  to a VAD-style summary, then parse (same metrics as VAD).

Experiment layout:
- Two methods: Arachne_v2, semSegRep
- Three fitness objectives: DISC, CONT, CONT2
- Several repetitions per experiment (whatever exists on disk)
- Weight count wXX is parsed from folder names (VAD often w26/52/105; UniAD may be w7/13/14, etc.)

Metrics extracted:
- plan_L2_1s, plan_L2_2s, plan_L2_3s (L2 error)
- plan_obj_box_col_1s, plan_obj_box_col_2s, plan_obj_box_col_3s (collision rate)
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
    Parse metrics from a VAD-style log or summary file.
    Supports plan_L2_1s:value lines and convert_uniad_to_vad_metrics.py summaries
    (e.g. "  plan_L2_1s: 0.5 (n=100)").
    """
    metrics = {}
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        for horizon in [1, 2, 3]:
            # Allow "plan_L2_1s:0.5" or "plan_L2_1s: 0.5 (n=...)" (optional space after colon)
            pattern = rf'plan_L2_{horizon}s:\s*([\d.e+-]+)'
            matches = re.findall(pattern, content)
            if matches:
                # UniAD converter summaries list each metric twice (avg_all / avg_valid).
                # Take the last match so we default to avg_valid (valid frames only).
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
    Count frames with collision from converter JSON (fut_valid_flag per VAD rules):
    valid frames where plan_obj_box_col_3s > 0.
    Matches the baseline 588-frame counting convention.
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

def count_vad_collision_from_json(json_path):
    """
    Collision stats from VAD open-loop JSON (no mean rescaling).

    Returns:
      (valid_frames, collision_frames, collision_events)
        - valid_frames: count with fut_valid_flag == True
        - collision_frames: valid frames with plan_obj_box_col_3s > 0 (at most one per frame)
        - collision_events: sum(plan_obj_box_col_3s * 6) (six-step collision events)
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        valid = 0
        cframes = 0
        events = 0.0
        for item in data:
            if not item.get("fut_valid_flag"):
                continue
            valid += 1
            v = item.get("plan_obj_box_col_3s")
            if not isinstance(v, (int, float)) or v < 0:
                continue
            if v > 0:
                cframes += 1
            events += float(v) * 6.0
        return valid, cframes, events
    except Exception:
        return None, None, None


def parse_experiment_name(folder_name):
    """Parse experiment folder name into configuration fields."""
    # Example: VAD_base_REP_VAL_Arachne_v2_DE_w26_p52_i50_es5_CONT_1
    # Or: VAD_base_REP_VAL_semSegRep_DE_w26_p52_i50_es5_CONT_1

    parts = folder_name.split('_')

    # Method name
    method = None
    if 'Arachne_v2' in folder_name:
        method = 'Arachne_v2'
    elif 'semSegRep' in folder_name:
        method = 'semSegRep'
    
    # Weight count
    weight_count = None
    for i, part in enumerate(parts):
        if part.startswith('w') and part[1:].isdigit():
            weight_count = int(part[1:])  # Numeric part after 'w'
            break

    # Fitness type (second-to-last token)
    fitness_type = None
    if len(parts) >= 2:
        fitness_type = parts[-2]
        if fitness_type not in ['DISC', 'CONT', 'CONT2']:
            fitness_type = None
    
    # Repetition index (last token)
    repetition = None
    if len(parts) >= 1:
        try:
            repetition = int(parts[-1])
        except ValueError:
            repetition = None
    
    return method, weight_count, fitness_type, repetition

def cohens_d(group1, group2, paired=False):
    """
    Compute Cohen's d effect size.

    Parameters:
    - group1, group2: two samples
    - paired: use paired-sample formula if True
    """
    if len(group1) == 0 or len(group2) == 0:
        return None
    
    if paired:
        # Paired Cohen's d
        if len(group1) != len(group2):
            return None
        diffs = group1 - group2
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1) if len(diffs) > 1 else 0
        if std_diff == 0:
            return None
        d = mean_diff / std_diff
    else:
        # Independent-samples Cohen's d
        mean1 = np.mean(group1)
        mean2 = np.mean(group2)
        std1 = np.std(group1, ddof=1) if len(group1) > 1 else 0
        std2 = np.std(group2, ddof=1) if len(group2) > 1 else 0

        # Pooled standard deviation
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
    Map Cohen's d to qualitative labels (group2 better when d > 0).
    Rules:
    - large_worse = d > 0.8        (group2 much better)
    - medium_worse = 0.2 <= d <= 0.8 (group2 moderately better)
    - small_worse = 0 < d < 0.2      (group2 slightly better)
    - small_better = -0.2 < d < 0   (group1 slightly better)
    - medium_better = -0.8 <= d <= -0.2 (group1 moderately better)
    - large_better = d < -0.8       (group1 much better)
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
    # CLI
    parser = argparse.ArgumentParser(description="Compute Cohen's d statistics")
    # NOTE: Only 3s metrics are kept; 1s/2s/3s time-horizon CLI options were removed
    parser.add_argument('--model-type', type=str, choices=['vad', 'uniad'], default='vad',
                        help='Model type: vad or uniad (default: vad)')
    args = parser.parse_args()

    # Experiment roots per model type
    # NOTE: Hard-coded data root for this machine's disk layout
    data_root = "/data1/deyun/E2ERep_Data"
    if args.model_type == 'vad':
        base_dirs = [
            str(Path(data_root) / "vad_base_Arachne_v2_DE_results"),
            str(Path(data_root) / "vad_base_semSegRep_DE_results"),
        ]
        folder_patterns = ['VAD_base_REP_VAL_*']
        def get_log_name(folder_name):
            return 'vad_base_rep_val.log'
    else:
        base_dirs = [
            str(Path(data_root) / "uniad_base_Arachne_v2_DE_results"),
            str(Path(data_root) / "uniad_base_semSegRep_DE_results"),
        ]
        folder_patterns = ['UniAD_base_REP_VAL_*', 'UniAD_tiny_REP_VAL_*']
        def get_log_name(folder_name):
            if 'UniAD_tiny' in folder_name or 'UniAD_tiny_REP' in folder_name:
                return 'uniad_tiny_rep_val.log'
            return 'uniad_base_rep_val.log'
    
    # Aggregated rows
    all_data = []
    missing_log = []  # Runs missing log/json
    parse_failed = []  # Unparseable folder names
    extract_failed = []  # Metrics could not be read
    success_count = 0  # Successfully parsed experiments
    
    print("=" * 80)
    print("Step 1: Extract metrics from all experiments")
    print("=" * 80)

    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Warning: {base_dir} not found")
            continue

        print(f"\nProcessing directory: {base_dir}")
        base_path = Path(base_dir)

        # Layout: <base_path>/(small|middle|large)/<EXP>/open_loop_eval/...
        exp_folders = []
        for pat in folder_patterns:
            for exp_folder in base_path.rglob(pat):
                if exp_folder.is_dir():
                    exp_folders.append(exp_folder)
        exp_folders = sorted(set(exp_folders))
        print(f"  Found {len(exp_folders)} experiment folders")

        # UniAD: convert JSON to VAD-style summary via convert_uniad_to_vad_metrics.py
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
                # VAD: collision counts from open_loop_eval/*.json (no mean rescaling)
                open_loop_dir = exp_folder / 'open_loop_eval'
                vad_json = json_file
                if not vad_json.exists() and open_loop_dir.exists():
                    all_json_candidates = sorted(open_loop_dir.glob("*.json"))
                    if len(all_json_candidates) == 1:
                        vad_json = all_json_candidates[0]
                if vad_json.exists() and metrics is not None:
                    _valid, _cframes, _events = count_vad_collision_from_json(vad_json)
                    if _cframes is not None:
                        metrics['collision_frame_count'] = _cframes
                    if _events is not None:
                        metrics['collision_event_count'] = float(_events)
            else:
                # UniAD: need JSON, then converter produces VAD-style summary for parsing
                if not json_file.exists():
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

            # Parse folder metadata
            method, weight_count, fitness_type, repetition = parse_experiment_name(exp_folder.name)
            if not all([method, weight_count, fitness_type, repetition]):
                parse_failed.append(str(exp_folder.relative_to(base_path)))
                continue
            if metrics is None:
                extract_failed.append(str(exp_folder.relative_to(base_path)))
                continue

            # Append row
            data_entry = {
                'method': method,
                'weight_count': weight_count,
                'fitness_type': fitness_type,
                'repetition': repetition,
                'folder': exp_folder.name,
                **metrics,
            }
            all_data.append(data_entry)
            success_count += 1

    # Report extraction summary
    print(f"\n" + "=" * 80)
    print("Data extraction summary")
    print("=" * 80)
    print(f"OK extracted: {success_count} experiments")
    print(f"Missing log files: {len(missing_log)} experiments")
    print(f"Failed to parse config: {len(parse_failed)} experiments")
    print(f"Failed to extract metrics: {len(extract_failed)} experiments")
    print(f"Total: {success_count + len(missing_log) + len(parse_failed) + len(extract_failed)} experiment folders")

    if missing_log:
        print(f"\nExperiments missing log files ({len(missing_log)}):")
        for folder in missing_log[:20]:  # Show at most 20
            print(f"  - {folder}")
        if len(missing_log) > 20:
            print(f"  ... {len(missing_log) - 20} more")

    if parse_failed:
        print(f"\nExperiments with unparsable config ({len(parse_failed)}):")
        for folder in parse_failed[:20]:  # Show at most 20
            print(f"  - {folder}")
        if len(parse_failed) > 20:
            print(f"  ... {len(parse_failed) - 20} more")

    if extract_failed:
        print(f"\nExperiments with metric extraction failure ({len(extract_failed)}):")
        for folder in extract_failed[:20]:  # Show at most 20
            print(f"  - {folder}")
        if len(extract_failed) > 20:
            print(f"  ... {len(extract_failed) - 20} more")
    
    print(f"\nExtracted data from {len(all_data)} experiments")

    if len(all_data) == 0:
        print("\n" + "=" * 80)
        print("Error: no experiment data extracted!")
        print("=" * 80)
        print("\nPlease check the missing-data details above.")
        print("Possible causes:")
        print("1. Incorrect experiment directory path")
        print("2. VAD: missing open_loop_eval/<model>_rep_val.log; UniAD: missing open_loop_eval/<model>_rep_val.json")
        print("3. Log files do not contain required metrics (plan_L2_*s or plan_obj_box_col_*s)")
        print("4. Experiment folder name format is incorrect; cannot parse config")
        print("\nScript stopped.")
        sys.exit(1)

    # Build DataFrame for analysis
    df = pd.DataFrame(all_data)
    # Weight and fitness lists from data (VAD w26/52/105 vs UniAD w7/13/14, etc.)
    weight_counts = sorted(df['weight_count'].unique().astype(int).tolist())
    # Fitness order for analyses/printing.
    fitness_types = ['CONT', 'CONT2', 'DISC']
    methods = ['Arachne_v2', 'semSegRep']
    weight_labels = [f"w{w}" for w in weight_counts]
    
    # Keep 3s metrics only (drop 1s/2s comparisons)
    metrics_to_analyze = [
        'plan_L2_3s',
        'plan_obj_box_col_3s',
    ]
    
    # Per-configuration means (3s only)
    print("\n" + "=" * 80)
    print("Per-config metric means (3s only)")
    print("=" * 80)
    config_cols = ['method', 'weight_count', 'fitness_type']
    mean_cols = ['plan_L2_3s', 'plan_obj_box_col_3s']
    config_means = df.groupby(config_cols)[mean_cols].mean().round(6)
    print("\n[Per-config means] (method, weight_count, fitness_type) -> metric means")
    for idx in config_means.index:
        method, weight_count, fitness_type = idx
        row = config_means.loc[idx]
        n = int(df[(df['method'] == method) & (df['weight_count'] == weight_count) & (df['fitness_type'] == fitness_type)].shape[0])
        print(
            f"  {method}, w{weight_count}, {fitness_type} (n={n}): "
            f"L2_3s={row['plan_L2_3s']:.6f}, col_3s={row['plan_obj_box_col_3s']:.6f}"
        )
    # L2 @ 3s means by configuration
    print("\n[L2 error 3s per-config averages]")
    l2_3s = df.groupby(config_cols)['plan_L2_3s'].agg(['mean', 'std', 'count']).round(6)
    for idx in l2_3s.index:
        method, weight_count, fitness_type = idx
        mean_val = l2_3s.at[idx, 'mean']
        std_val = l2_3s.at[idx, 'std']
        count_val = int(l2_3s.at[idx, 'count'])
        std_str = f"{std_val:.6f}" if pd.notna(std_val) else "N/A"
        print(f"  {method}, w{weight_count}, {fitness_type}: mean={mean_val:.6f}, std={std_str}, n={count_val}")
    # Collision @ 3s means; UniAD uses collision-frame counts, VAD uses rate
    print("\n[Collision 3s per-config averages]")
    collision_col = 'collision_frame_count' if args.model_type in ('uniad', 'vad') else 'plan_obj_box_col_3s'
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

    # Boxplots for L2/Collision @ 3s across repetitions with un-repaired baselines
    # VAD and UniAD baselines differ (both valid-frame statistics)
    if args.model_type == 'vad':
        # VAD baseline from original open-loop eval
        # Legacy code scaled plan_obj_box_col_3s mean * 6121 * 6 to #colls, which
        # mis-represented valid frames with collisions; now taken from baseline JSON.
        BASELINE_L2_3S = 1.4262655224607823
        # baseline/VAD/vad_base_baseline_b2d_infos_val_partB_25clips.json:
        # - collision_frames (valid & plan_obj_box_col_3s>0) = 80
        # - collision_events (sum plan_obj_box_col_3s*6) ≈ 85
        BASELINE_COLLISION_FRAME_COUNT = 80
        BASELINE_COLLISION_EVENT_COUNT = 85
    else:
        # UniAD baseline from uniad_base_baseline_b2d_infos_val_partB_25clips.json via
        # baseline/convert_uniad_to_vad_metrics.py (VAD rules):
        # - valid frames = fut_valid_flag (VAD: full_future and mask_sum > 0), n=6371
        # - BASELINE_L2_3S = plan_L2_3s (avg_valid)
        # - BASELINE_COLLISION_3S = plan_obj_box_col_3s (avg_valid): per-frame non-neg
        #   means of plan_obj_box_col_1s/2s/3s averaged over valid frames
        # - BASELINE_COLLISION_FRAME_COUNT = collision frames under VAD rules (legend)
        BASELINE_L2_3S = 1.2513238752833544
        BASELINE_COLLISION_3S = 0.014457873542816614
        BASELINE_COLLISION_FRAME_COUNT = 588
    if HAS_MATPLOTLIB:
        # Deterministic boxplot order:
        #  semSegRep first (9 boxes) then Arachne_v2 (9 boxes)
        #  Within each method: fitness CONT -> CONT2 -> DISC
        #  Then weight_count ascending (e.g. w7 -> w13 -> w26)
        config_order = []
        for method in ['semSegRep', 'Arachne_v2']:
            # Explicit fitness order for each method strip
            for fitness_type in ['CONT', 'CONT2', 'DISC']:
                for weight_count in weight_counts:
                    key = (method, weight_count, fitness_type)
                    if key in config_means.index:
                        config_order.append(key)
        # X-axis layout (2×3×3 hierarchy):
        # - Inner ticks: weight tier only (Low/Med/High)
        # - Mid brackets: every 3 boxes (same method + fitness) label the fit term
        # - Big brackets: every 9 boxes (same method) label the approach / loss
        #
        # Avoids repeating full config strings on every tick.
        box_level_labels = []
        mid_group_specs = []  # (start, end, mid_label_mathtext)  e.g. "$fit_{ER}$"
        big_group_specs = []  # (start, end, big_label_mathtext)  e.g. "$FL_{naive}$"
        # Map (method, fitness_type, weight_count) to LaTeX/mathtext-safe labels
        loss_map = {
            "Arachne_v2": r"FL_{adv}",
            "semSegRep": r"FL_{naive}",
        }
        fit_map = {
            "DISC": r"fit_{CA}",
            "CONT": r"fit_{ER}",
            "CONT2": r"fit_{ERC}",
        }
        # Weight tier labels:
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

        prev_mid_key = None
        cur_mid_start = None
        cur_mid_label = None
        prev_big_key = None
        cur_big_start = None
        cur_big_label = None

        for idx, (method, weight_count, fitness_type) in enumerate(config_order):
            loss_str = loss_map.get(method, method)
            fit_str = fit_map.get(fitness_type, fitness_type)
            mid_key = (loss_str, fit_str)  # (big, mid)
            big_key = loss_str
            try:
                w_int = int(weight_count)
            except Exception:
                w_int = None
            level_str = level_map.get(w_int, f"w{weight_count}")
            box_level_labels.append(level_str)

            # mid group: (loss, fit)
            if prev_mid_key is None:
                prev_mid_key = mid_key
                cur_mid_start = idx
                cur_mid_label = rf"${fit_str}$"
            elif mid_key != prev_mid_key:
                mid_group_specs.append((cur_mid_start, idx - 1, cur_mid_label))
                prev_mid_key = mid_key
                cur_mid_start = idx
                cur_mid_label = rf"${fit_str}$"

            # big group: loss only
            if prev_big_key is None:
                prev_big_key = big_key
                cur_big_start = idx
                cur_big_label = rf"${loss_str}$"
            elif big_key != prev_big_key:
                big_group_specs.append((cur_big_start, idx - 1, cur_big_label))
                prev_big_key = big_key
                cur_big_start = idx
                cur_big_label = rf"${loss_str}$"

        # flush last groups
        if cur_mid_start is not None:
            mid_group_specs.append((cur_mid_start, len(config_order) - 1, cur_mid_label))
        if cur_big_start is not None:
            big_group_specs.append((cur_big_start, len(config_order) - 1, cur_big_label))
        l2_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['plan_L2_3s'].dropna().values
                   for (m, w, f) in config_order]
        # Collision plot:
        # - VAD: collision_frames from JSON (at most once per frame)
        # - UniAD: collision_frame_count from converter JSON (same convention)
        if args.model_type == 'vad':
            col_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['collision_frame_count'].dropna().values
                       for (m, w, f) in config_order]
            baseline_collision_plot = BASELINE_COLLISION_FRAME_COUNT
        else:
            col_data = [df[(df['method'] == m) & (df['weight_count'] == w) & (df['fitness_type'] == f)]['collision_frame_count'].dropna().values
                       for (m, w, f) in config_order]
            baseline_collision_plot = BASELINE_COLLISION_FRAME_COUNT
        out_dir = Path(__file__).parent
        def _draw_bracket(ax, x0, x1, y_axes, text, text_y_offset=0.02, color="#555555", lw=1.2):
            # x0/x1: data coord (boxplot positions), y in axes coord
            trans = ax.get_xaxis_transform()
            ax.plot([x0, x0, x1, x1], [y_axes - 0.02, y_axes, y_axes, y_axes - 0.02],
                    transform=trans, color=color, linewidth=lw, clip_on=False)
            ax.text((x0 + x1) / 2.0, y_axes + text_y_offset, text,
                    transform=trans, ha="center", va="bottom", color=color, clip_on=False)

        def add_hierarchical_labels(ax):
            # Draw mid (fit) and big (loss) bracket annotations above the axes
            # Mid: groups of 3 boxes sharing method+fitness
            for s, e, label in mid_group_specs:
                x0 = s + 1.0
                x1 = e + 1.0
                # Raise y slightly so brackets are not clipped
                _draw_bracket(ax, x0, x1, y_axes=1.03, text=label, text_y_offset=0.01, color="#666666", lw=1.1)
            # Big: every 9 boxes share the method
            for s, e, label in big_group_specs:
                x0 = s + 1.0
                x1 = e + 1.0
                _draw_bracket(ax, x0, x1, y_axes=1.14, text=label, text_y_offset=0.01, color="#333333", lw=1.4)
            # Light vertical separators between big groups
            for s, e, _ in big_group_specs[:-1]:
                x_sep = (e + 1) + 0.5
                ax.axvline(x=x_sep, color="#dddddd", linewidth=1.0, zorder=0)

        # Shared figure size/margins so stacked plots align (publication-friendly height)
        FIG_SIZE = (14, 5)
        SUBPLOT_LEFT, SUBPLOT_RIGHT = 0.06, 0.98
        # Reserve top margin for brackets + external legend
        SUBPLOT_BOTTOM, SUBPLOT_TOP = 0.20, 0.74
        # Fixed axes box [left, bottom, width, height] shared by both figures so
        # bbox_inches='tight' still keeps data areas aligned.
        # Extra top room prevents legend colliding with bracket labels.
        AX_POS = [0.06, 0.18, 0.92, 0.54]
        SAVE_PAD_INCHES = 0.02
        LEGEND_ANCHOR_Y = 1.00
        # Global font size for axes, ticks, legend (paper-ready)
        plt.rcParams['font.size'] = 20
        # L2 @ 3s boxplot (18 configs) + baseline horizontal line
        fig1, ax1 = plt.subplots(figsize=FIG_SIZE)
        fig1.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP)
        ax1.set_position(AX_POS)
        # Blank x tick labels; tiers encoded by color + legend
        bp1 = ax1.boxplot(l2_data, labels=[""] * len(box_level_labels), patch_artist=True, showfliers=False)
        ax1.axhline(
            y=BASELINE_L2_3S,
            color='red',
            linestyle='--',
            linewidth=1.5,
            # L2_Err(M_{orig}, D_{test})
            label=rf"$\mu_{{\mathrm{{L2\_Err}}}}(M_{{\mathrm{{orig}}}}, D_{{\mathrm{{test}}}})$ = {BASELINE_L2_3S:.4f}",
        )
        # Y axis shows distribution of mean metric → label μ_{L2_Err}
        ax1.set_ylabel(r"$\mu_{\mathrm{L2\_Err}}$")
        ax1.tick_params(axis='x', rotation=0, length=0)
        add_hierarchical_labels(ax1)
        # Color policy:
        # - Face color encodes weight tier (Low/Med/High) only
        # - Big/mid structure from brackets + separators (not extra colors)
        level_colors = {
            "Low": "#66c2a5",   # green
            "Med": "#fc8d62",   # orange
            "High": "#8da0cb",  # purple/blue
        }
        for i, patch in enumerate(bp1['boxes']):
            lvl = box_level_labels[i]
            patch.set_facecolor(level_colors.get(lvl, "#cccccc"))
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.2)
        for median in bp1['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        # Legend above the figure to avoid covering brackets/data
        try:
            from matplotlib.patches import Patch
            handles, _labels = ax1.get_legend_handles_labels()
            handles += [
                Patch(facecolor=level_colors["Low"], edgecolor="#333333", label="Low"),
                Patch(facecolor=level_colors["Med"], edgecolor="#333333", label="Med"),
                Patch(facecolor=level_colors["High"], edgecolor="#333333", label="High"),
            ]
            fig1.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
                bbox_transform=fig1.transFigure,
                ncol=4,
                frameon=True,
            )
        except Exception:
            fig1.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
                bbox_transform=fig1.transFigure,
                ncol=4,
                frameon=True,
            )
        l2_pdf = out_dir / f"l2_3s_boxplot_{args.model_type}.pdf"
        fig1.savefig(l2_pdf, bbox_inches='tight', pad_inches=SAVE_PAD_INCHES)
        plt.close(fig1)
        # Collision @ 3s plot + baseline (VAD: event counts; UniAD: collision frames)
        fig2, ax2 = plt.subplots(figsize=FIG_SIZE)
        fig2.subplots_adjust(left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP)
        ax2.set_position(AX_POS)
        bp2 = ax2.boxplot(col_data, labels=[""] * len(box_level_labels), patch_artist=True, showfliers=False)
        ax2.axhline(
            y=baseline_collision_plot,
            color='red',
            linestyle='--',
            linewidth=1.5,
            # Prefix '#' as plain text; colls(...) rendered with mathtext
            label="#" + rf"$\mathrm{{colls}}(M_{{orig}}, D_{{test}})$ = {baseline_collision_plot:.0f}",
        )
        ax2.set_ylabel('#colls')
        ax2.tick_params(axis='x', rotation=0, length=0)
        add_hierarchical_labels(ax2)
        for i, patch in enumerate(bp2['boxes']):
            lvl = box_level_labels[i]
            patch.set_facecolor(level_colors.get(lvl, "#cccccc"))
            patch.set_alpha(0.85)
            patch.set_edgecolor("#333333")
            patch.set_linewidth(1.2)
        for median in bp2['medians']:
            median.set_color('#333333')
            median.set_linewidth(1.5)
        try:
            from matplotlib.patches import Patch
            handles, labels = ax2.get_legend_handles_labels()
            handles += [
                Patch(facecolor=level_colors["Low"], edgecolor="#333333", label="Low"),
                Patch(facecolor=level_colors["Med"], edgecolor="#333333", label="Med"),
                Patch(facecolor=level_colors["High"], edgecolor="#333333", label="High"),
            ]
            fig2.legend(
                handles=handles,
                loc="upper center",
                bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
                bbox_transform=fig2.transFigure,
                ncol=4,
                frameon=True,
            )
        except Exception:
            fig2.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, LEGEND_ANCHOR_Y),
                bbox_transform=fig2.transFigure,
                ncol=4,
                frameon=True,
            )
        col_pdf = out_dir / f"collision_3s_boxplot_{args.model_type}.pdf"
        fig2.savefig(col_pdf, bbox_inches='tight', pad_inches=SAVE_PAD_INCHES)
        plt.close(fig2)
        print(f"\nBoxplots saved: {l2_pdf}, {col_pdf}")
    else:
        print("\n(matplotlib not installed; skipping boxplots)")

    print("\n" + "=" * 80)
    print("Step 2: Compute Cohen's d statistics")
    print("=" * 80)
    
    results = []
    
    # Comparison 1: methods (Arachne_v2 vs semSegRep), paired on shared configs
    print("\n--- Comparison 1: methods (Arachne_v2 vs semSegRep) ---")
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
                print(f"    Vector dim: {len(m1_array)}x1; order: {order_str}")
                print(f"    {method1} vector: {m1_array}")
                print(f"    {method2} vector: {m2_array}")
    
    # Comparison 2: fitness objectives
    print("\n--- Comparison 2: fitness functions ---")
    for metric in metrics_to_analyze:
        # Paired grid: each cell is (method, weight_count) with all three fitness values
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
            print(f"    Vector dim: {len(base_positions)}x1; order: {order_str}")
            for ft in fitness_types:
                print(f"    {ft} vector: {fitness_vectors[ft]}")

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

    # Comparison 3: weight counts
    print("\n--- Comparison 3: weight counts ---")
    for metric in metrics_to_analyze:
        # Paired grid: each cell is (method, fitness_type) with every weight count present
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
            print(f"    Vector dim: {len(base_positions)}x1; order: {order_str}")
            for label in weight_labels:
                if label in weight_vectors:
                    print(f"    {label} vector: {weight_vectors[label]}")

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

    # Results table
    results_df = pd.DataFrame(results)
    
    # Pretty-print summary
    print("\n" + "=" * 80)
    print("Step 3: Summary report")
    print("=" * 80)
    
    # Group by comparison type and metric
    print("\nCohen's d statistics summary\n")

    for comparison_type in ['Method', 'Fitness', 'Weight']:
        print(f"\n{'=' * 80}")
        print(f"Comparison type: {comparison_type}")
        print(f"{'=' * 80}\n")

        subset = results_df[results_df['comparison_type'] == comparison_type]

        for metric in metrics_to_analyze:
            metric_subset = subset[subset['metric'] == metric]
            if len(metric_subset) > 0:
                if comparison_type == 'Fitness':
                    # Fitness: 3×3 matrix (same ordering as Step 2)
                    fitness_types = ['CONT', 'CONT2', 'DISC']
                    # Build matrix
                    matrix = {}
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            effect_size = interpret_cohens_d(d)
                            key = (row['group1'], row['group2'])
                            matrix[key] = effect_size
                    
                    print(f"Metric: {metric}; CONT vs CONT2 vs DISC")
                    print("    " + " " * 12 + "CONT" + " " * 12 + "CONT2" + " " * 12 + "DISC")
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
                                    row_str += "identical".center(16)  # Truly identical or symmetric diffs
                        print(row_str)
                elif comparison_type == 'Weight':
                    # Weight: N×N matrix (N = number of weight tiers)
                    # Prefer sorted weight_counts, else lexical fallback
                    weight_names = list(weight_labels)
                    if not weight_names:
                        # Fallback: infer names from results
                        names = set(metric_subset['group1'].dropna().tolist()) | set(metric_subset['group2'].dropna().tolist())
                        def _w_key(x):
                            m = re.search(r'(\d+)', str(x))
                            return int(m.group(1)) if m else 10**9
                        weight_names = sorted(names, key=_w_key)
                    # Build matrix
                    matrix = {}
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            effect_size = interpret_cohens_d(d)
                            key = (row['group1'], row['group2'])
                            matrix[key] = effect_size
                    
                    header = "    " + " " * 12 + "".join([f"{w:>16s}" for w in weight_names])
                    print(f"Metric: {metric}; " + " vs ".join(weight_names))
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
                    # Method: single-line summary rows
                    for _, row in metric_subset.iterrows():
                        if pd.notna(row['cohens_d']):
                            d = row['cohens_d']
                            # Map d to qualitative label
                            effect_size = interpret_cohens_d(d)
                            
                            print(f"Metric: {metric}; {row['group1']} vs {row['group2']}: {effect_size}")
    
    # NOTE: Only 3s metrics remain; cross-horizon (1s/2s/3s) comparisons were removed.
    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)

if __name__ == '__main__':
    main()

