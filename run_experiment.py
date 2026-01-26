#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simplified automated experiment script for repair and evaluation.
"""

import sys
import subprocess
from pathlib import Path
import argparse
import re
import os
import shlex


def str2bool(v):
    """
    Parse common string booleans for argparse.
    Accepts: true/false, 1/0, yes/no, y/n, t/f (case-insensitive).
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v!r}")


def get_model_paths(model_name, model_type, repo_root):
    """Get Bench2DriveZoo model config/checkpoint paths."""
    b2d_root = repo_root / "Bench2DriveZoo"
    
    if model_type == "VAD":
        if model_name == "VAD_base":
            return {
                "config": b2d_root / "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py",
                "checkpoint": b2d_root / "ckpts/vad_b2d_base.pth",
            }
        # VAD_tiny in this repo refers to the tiny B2D checkpoint; it still uses the B2D config
        return {
            "config": b2d_root / "adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py",
            "checkpoint": b2d_root / "ckpts/vad_b2d_tiny.pth",
        }
    elif model_type == "UniAD":
        if model_name == "UniAD_base":
            return {
                "config": b2d_root / "adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py",
                "checkpoint": b2d_root / "ckpts/uniad_base_b2d.pth",  # Fixed: uniad_base_b2d.pth (not uniad_b2d_base.pth)
            }
        elif model_name == "UniAD_tiny":
            return {
                "config": b2d_root / "adzoo/uniad/configs/stage2_e2e/tiny_e2e_b2d.py",
                "checkpoint": b2d_root / "ckpts/uniad_tiny_b2d.pth",
            }
        else:
            raise ValueError(f"Unknown UniAD name: {model_name}")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_vad_paths(vad_name, repo_root):
    """Legacy function for backward compatibility."""
    return get_model_paths(vad_name, "VAD", repo_root)


def generate_experiment_name(model_name, rep_method, search_algo, search_algo_params, 
                             fitness, run_idx, base_dir, time_horizon=3):
    """Generate experiment directory name."""
    w = search_algo_params['num_weights_to_repair']
    p = search_algo_params['num_particles']
    i = search_algo_params['num_iterations']
    
    # Add early stop patience to search params if set
    early_stop = search_algo_params.get('early_stop_patience')
    if early_stop is not None and early_stop > 0:
        search_params = f"w{w}_p{p}_i{i}_es{early_stop}"
    else:
        search_params = f"w{w}_p{p}_i{i}"
    
    if fitness == 'continuous':
        fitness_str = 'CONT'
    elif fitness == 'continuous2':
        fitness_str = 'CONT2'
    else:
        fitness_str = 'DISC'
    
    # Add time_horizon to experiment name (e.g., t1s, t2s, t3s)
    time_horizon_str = f"t{time_horizon}s"
    
    # Auto-detect run_idx
    if run_idx is None:
        pattern = f"{model_name}_REP_VAL_{time_horizon_str}_{rep_method}_{search_algo}_{search_params}_{fitness_str}_([0-9]+)"
        max_idx = 0
        if base_dir.exists():
            for item in base_dir.iterdir():
                if item.is_dir():
                    match = re.match(pattern, item.name)
                    if match:
                        max_idx = max(max_idx, int(match.group(1)))
        run_idx = max_idx + 1
    
    return f"{model_name}_REP_VAL_{time_horizon_str}_{rep_method}_{search_algo}_{search_params}_{fitness_str}_{run_idx}"


def generate_baseline_json(model_name, model_type, repair_dataset, base_dir, repo_root,
                           cuda_device='0', regenerate=False,
                           target_filename=None, cfg_options=None,
                           occ_output_dir=None):
    """
    Generate baseline JSON by evaluating original model on repair dataset.
    
    Args:
        model_name: Model name (VAD_base, VAD_tiny, UniAD_base, UniAD_tiny)
        model_type: Model type ('VAD' or 'UniAD')
        repair_dataset: Path to repair dataset PKL file
        base_dir: Base experiments directory (baseline JSON will be stored here)
        repo_root: Repository root directory
        cuda_device: CUDA device ID
        regenerate: If True, regenerate even if file exists
    Returns:
        Path to baseline JSON file, or None if failed
    """
    model_paths = get_model_paths(model_name, model_type, repo_root)
    model_lower = model_name.lower()
    
    # Decide baseline JSON + log file paths
    if target_filename is not None:
        baseline_json = (base_dir / target_filename).resolve()
        log_stem = Path(target_filename).stem
        log_file = (base_dir / f"{log_stem}.log").resolve()
    else:
        dataset_name = Path(repair_dataset).stem
        baseline_json = (base_dir / f"{model_lower}_baseline_{dataset_name}.json").resolve()
        log_file = (base_dir / f"{model_lower}_baseline_{dataset_name}.log").resolve()
    
    # Check if baseline already exists
    # If file exists, always reuse it (ignore regenerate flag to avoid unnecessary recomputation)
    if baseline_json.exists():
        print(f"\nBaseline JSON already exists: {baseline_json}")
        print("Reusing existing baseline (regenerate flag is ignored if file exists).")
        return baseline_json
    
    print("\n" + "="*80)
    print("Generating Baseline JSON")
    print("="*80)
    print(f"Repair dataset: {repair_dataset}")
    print(f"Output: {baseline_json}")
    
    # Ensure parent exists even when running child process from a different cwd
    baseline_json.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Build cfg-options string
    if cfg_options is None:
        cfg_options = f"data.test.ann_file='{repair_dataset}'"

    if model_type == "VAD":
        cmd = [
            f"CUDA_VISIBLE_DEVICES={cuda_device}",
            sys.executable,
            '-u',
            str(repo_root / "Bench2DriveZoo" / "adzoo" / "vad" / "test.py"),
            str(model_paths['config']),
            str(model_paths['checkpoint']),
            '--cfg-options', cfg_options,
            '--launcher', 'none',
            '--eval', 'bbox',
            '--tmpdir', 'tmp',
            '--eval-options', 'jsonfile_prefix=None',
            '--collect-data',
            '--data-output', str(baseline_json),
        ]
    elif model_type == "UniAD":
        cmd = [
            f"CUDA_VISIBLE_DEVICES={cuda_device}",
            "torchrun",
            "--nproc_per_node=1",
            str(repo_root / "Bench2DriveZoo" / "adzoo" / "uniad" / "test.py"),
            str(model_paths['config']),
            str(model_paths['checkpoint']),
            "--launcher", "pytorch",
            "--cfg-options", cfg_options,
            "--eval", "bbox",
            "--collect-data",
            "--data-output", str(baseline_json),
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if occ_output_dir:
        cmd.extend(['--occ-output-dir', str(occ_output_dir)])
    
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    
    b2d_root = repo_root / "Bench2DriveZoo"
    with open(log_file, 'w', buffering=1) as f:  # Line buffering for real-time output
        # Also record the exact command (including cfg-options) into the log for debugging.
        f.write(f"CMD: {cmd_str}\n")
        f.flush()
        result = subprocess.run(cmd_str, shell=True, cwd=b2d_root,
                              stdout=f, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    if result.returncode != 0 or not baseline_json.exists():
        print(f"ERROR: Baseline JSON generation failed. Check {log_file}")
        return None
    
    print(f"Baseline JSON generated: {baseline_json}")
    return baseline_json


def run_repair(model_name, model_type, baseline_json, exp_dir, repo_root, 
              alpha=0.5, layers=None,
              fitness_type='continuous', num_particles=200, num_iterations=100, 
              num_weights_to_repair=100, rep_method='Arachne_v1', early_stop_patience=None,
              search_algo='PSO', cuda_device='0', time_horizon=3, collision_num_workers=None):
    """Run repair process."""
    print("\n" + "="*80)
    print("Running Repair")
    print("="*80)
    
    model_paths = get_model_paths(model_name, model_type, repo_root)
    repair_dir = exp_dir / "repair"
    repair_dir.mkdir(exist_ok=True)
    output_dir = (repair_dir / "repair_output").resolve()
    
    # Set default layers based on model type
    if layers is None:
        if model_type == "VAD":
            layers = 'pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2'
        elif model_type == "UniAD":
            layers = 'planning_head.reg_branch.0'  # Only the 256x256 hidden layer
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    # Choose repair script based on model type
    if model_type == "VAD":
        repair_script = repo_root / "repair" / "repair_ego_fut_decoder_arachne.py"
        repaired_model_name = "VAD_repaired_both_layers.pth"
    elif model_type == "UniAD":
        repair_script = repo_root / "repair_uniad" / "repair_planning_head_arachne.py"
        repaired_model_name = "repaired_model.pth"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    cmd = [
        f"CUDA_VISIBLE_DEVICES={cuda_device}",
        sys.executable,
        str(repair_script),
        '--config', str(model_paths['config']),
        '--checkpoint', str(model_paths['checkpoint']),
        '--json', str(baseline_json),
        '--alpha', str(alpha),
        '--rep-method', rep_method,
        '--search-algo', search_algo,
        '--layers'] + layers.split() + [
        '--fitness-type', fitness_type,
        '--num-particles', str(num_particles),
        '--num-iterations', str(num_iterations),
        '--num-weights-to-repair', str(num_weights_to_repair),
        '--output-dir', str(output_dir),
    ]
    
    # Add early stop patience if enabled
    if early_stop_patience is not None and int(early_stop_patience) > 0:
        cmd.extend(['--early-stop-patience', str(early_stop_patience)])
    
    # Add time horizon parameter
    cmd.extend(['--time-horizon', str(time_horizon)])

    # Optional: collision worker parallelism
    if collision_num_workers is not None:
        cmd.extend(['--collision-num-workers', str(collision_num_workers)])
    
    # Always use cached evaluation for VAD repair
    if model_type == "VAD":
        cmd.append('--use-cached-eval')
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Using CUDA device: {cuda_device}")
    log_file = (repair_dir / "repair.log").resolve()

    # Check if repaired model already exists
    repaired_model = (output_dir / repaired_model_name).resolve()
    if repaired_model.exists():
        print(f"Repaired model already exists, reusing: {repaired_model}")
        return repaired_model
    
    # Use tee command to write to both file and terminal
    cmd_with_tee = f"({' '.join(cmd)}) | tee {shlex.quote(str(log_file))}"
    
    result = subprocess.run(cmd_with_tee, shell=True, cwd=repo_root, 
                          stderr=subprocess.STDOUT, text=True)
    
    repaired_model = (output_dir / repaired_model_name).resolve()
    
    # Check if repaired model exists (more reliable than return code)
    # Some scripts may exit with non-zero code due to display/X server errors
    # but still successfully generate the model file
    if not repaired_model.exists():
        if result.returncode != 0:
            print(f"ERROR: Repair failed (exit code {result.returncode}). Check {log_file}")
        else:
            print(f"ERROR: Repaired model not found at {repaired_model}")
        return False
    
    # Warn if exit code is non-zero but model exists (likely a non-critical error)
    if result.returncode != 0:
        print(f"WARNING: Repair script exited with code {result.returncode}, but model file exists.")
        print(f"  This may be due to display/X server errors (e.g., XIO errors).")
        print(f"  Continuing with evaluation. Check {log_file} for details.")
    
    print(f"Repaired model: {repaired_model}")
    return repaired_model


def run_evaluation(model_name, model_type, repaired_model, eval_dataset, exp_dir, repo_root,
                   cuda_device='0', use_tmpfs_data=False, occ_output_dir=None):
    """Run evaluation process."""
    print("\n" + "="*80)
    print("Running Open-loop Evaluation")
    print("="*80)
    
    model_paths = get_model_paths(model_name, model_type, repo_root)
    # Store open-loop results under exp_dir/open_loop_eval
    legacy_eval_dir = exp_dir / "evaluation"
    eval_dir = exp_dir / "open_loop_eval"
    if legacy_eval_dir.exists() and not eval_dir.exists():
        legacy_eval_dir.rename(eval_dir)
    eval_dir.mkdir(exist_ok=True)
    model_lower = model_name.lower()
    output_json = (eval_dir / f"{model_lower}_rep_val.json").resolve()
    
    # Build cfg-options string
    ann_file = str(eval_dataset)
    if use_tmpfs_data:
        tmpfs_base = "/mnt/bench2drive_ram"
        if model_type == "VAD":
            cfg_options = " ".join([
                f"data_root='{tmpfs_base}/bench2drive'",
                f"info_root='{tmpfs_base}/infos'",
                f"map_root='{tmpfs_base}/bench2drive/maps'",
                f"map_file='{tmpfs_base}/b2d_map_infos.pkl'",
                f"data.test.ann_file='{ann_file}'",
            ])
        else:  # UniAD
            cfg_options = " ".join([
                f"data.test.data_root='{tmpfs_base}/bench2drive'",
                f"data.test.map_file='{tmpfs_base}/infos/b2d_map_infos.pkl'",
                f"data.test.ann_file='{ann_file}'",
            ])
        print("\nUsing tmpfs dataset for evaluation:")
        print(f"  tmpfs_base={tmpfs_base}")
    else:
        cfg_options = f"data.test.ann_file='{ann_file}'"

    if model_type == "VAD":
        cmd = [
            f"CUDA_VISIBLE_DEVICES={cuda_device}",
            sys.executable,
            '-u',
            str(repo_root / "Bench2DriveZoo" / "adzoo" / "vad" / "test.py"),
            str(model_paths['config']),
            str(repaired_model),
            '--cfg-options', cfg_options,
            '--launcher', 'none',
            '--eval', 'bbox',
            '--tmpdir', 'tmp',
            '--eval-options', 'jsonfile_prefix=None',
            '--collect-data',
            '--data-output', str(output_json),
        ]
    elif model_type == "UniAD":
        cmd = [
            f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            f"CUDA_VISIBLE_DEVICES={cuda_device}",
            "torchrun",
            "--nproc_per_node=1",
            str(repo_root / "Bench2DriveZoo" / "adzoo" / "uniad" / "test.py"),
            str(model_paths['config']),
            str(repaired_model),
            "--launcher", "pytorch",
            "--cfg-options", cfg_options,
            "--eval", "bbox",
            "--collect-data",
            "--data-output", str(output_json),
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if occ_output_dir:
        cmd.extend(['--occ-output-dir', str(occ_output_dir)])
    
    cmd_str = ' '.join(cmd)
    print(f"Command: {cmd_str}")
    log_file = (eval_dir / f"{model_lower}_rep_val.log").resolve()
    
    b2d_root = repo_root / "Bench2DriveZoo"
    with open(log_file, 'w', buffering=1) as f:  # Line buffering for real-time output
        # Record the exact command (including cfg-options) into the log.
        f.write(f"CMD: {cmd_str}\n")
        f.flush()
        result = subprocess.run(cmd_str, shell=True, cwd=b2d_root,
                              stdout=f, stderr=subprocess.STDOUT, text=True, bufsize=1)
    
    if result.returncode != 0:
        print(f"ERROR: Evaluation failed. Check {log_file}")
        return False
    
    print(f"Evaluation results: {output_json}")
    return True


def run_closed_loop_evaluation(model_name, model_type, repaired_model, exp_dir, repo_root, args):
    """
    Scheme B: call a dedicated parameterized closed-loop shell script.

    Outputs:
      - checkpoint json: <exp_dir>/closed_loop_eval/closed_loop_eval.json
      - saved frames/metadata: under <exp_dir>/closed_loop_eval/...
    """
    print("\n" + "="*80)
    print("Running Closed-loop Evaluation")
    print("="*80)

    model_paths = get_model_paths(model_name, model_type, repo_root)
    cl_dir = exp_dir / "closed_loop_eval"
    cl_dir.mkdir(exist_ok=True)

    # Use fast or super-fast version if requested
    if args.closed_loop_super_fast and args.closed_loop_fast:
        print("ERROR: --closed-loop-fast and --closed-loop-super-fast are mutually exclusive")
        return False
    
    if args.closed_loop_super_fast:
        # Super-fast version: use super-fast config and super-fast agent
        super_fast_config = vad_paths['config'].parent / vad_paths['config'].name.replace('.py', '_super_fast.py')
        if not super_fast_config.exists():
            print(f"ERROR: Super-fast config not found: {super_fast_config}")
            return False
        team_config = f"{super_fast_config.resolve()}+{repaired_model}"
        team_agent = (repo_root / "Bench2DriveZoo/team_code/vad_b2d_agent_super_fast.py").resolve()
        if not team_agent.exists():
            print(f"ERROR: Super-fast agent not found: {team_agent}")
            return False
        print(f"Using super-fast version: agent={team_agent.name}, config={super_fast_config.name}")
    elif args.closed_loop_fast:
        # Fast version: use fast config and fast agent
        fast_config = vad_paths['config'].parent / vad_paths['config'].name.replace('.py', '_fast.py')
        if not fast_config.exists():
            print(f"ERROR: Fast config not found: {fast_config}")
            return False
        team_config = f"{fast_config.resolve()}+{repaired_model}"
        team_agent = (repo_root / "Bench2DriveZoo/team_code/vad_b2d_agent_fast.py").resolve()
        if not team_agent.exists():
            print(f"ERROR: Fast agent not found: {team_agent}")
            return False
        print(f"Using fast version: agent={team_agent.name}, config={fast_config.name}")
    else:
        # Normal version
        team_config = f"{model_paths['config'].resolve()}+{repaired_model}"
        team_agent = Path(args.closed_loop_team_agent)
        if not team_agent.is_absolute():
            team_agent = repo_root / team_agent
        team_agent = team_agent.resolve()
        if not team_agent.exists():
            print(f"ERROR: Closed-loop TEAM_AGENT not found: {team_agent}")
            return False

    checkpoint_json = (cl_dir / "closed_loop_eval.json").resolve()
    save_path = str(cl_dir.resolve())
    log_file = cl_dir / "closed_loop_eval.log"

    # Use a dedicated parameterized script (do not modify run_evaluation_debug.sh)
    script = repo_root / "leaderboard" / "scripts" / "run_closed_loop_eval.sh"
    if not script.exists():
        print(f"ERROR: Closed-loop script not found: {script}")
        return False

    env = os.environ.copy()
    if args.carla_root:
        env["CARLA_ROOT"] = str(Path(args.carla_root).expanduser())
    if not env.get("CARLA_ROOT"):
        print("ERROR: CARLA_ROOT is not set. Please export CARLA_ROOT or pass --carla-root.")
        return False

    cmd = [
        "bash",
        str(script),
        str(args.closed_loop_port),
        str(args.closed_loop_tm_port),
        str(args.closed_loop_is_bench2drive),
        str(routes_path),
        str(team_agent),
        team_config,
        str(checkpoint_json),
        save_path,
        str(args.closed_loop_planner_type),
        str(args.closed_loop_gpu_rank),
    ]

    print(f"Command: {' '.join(cmd)}")
    print(f"Checkpoint JSON: {checkpoint_json}")
    print(f"Save path: {save_path}")

    with open(log_file, "w", buffering=1) as f:
        result = subprocess.run(cmd, cwd=repo_root, env=env, stdout=f, stderr=subprocess.STDOUT, text=True)

    if checkpoint_json.exists():
        if result.returncode != 0:
            print(f"WARNING: Closed-loop evaluator exited with code {result.returncode}, but checkpoint exists.")
            print(f"  Check log: {log_file}")
        else:
            print(f"Closed-loop evaluation results: {checkpoint_json}")
        return True

    print(f"ERROR: Closed-loop evaluation failed. Check {log_file}")
    return False


def run_single_experiment(run_idx, model_name, model_type, repair_dataset, eval_dataset, 
                          base_dir, repo_root, args):
    """Run a single experiment."""
    # Helper to build cfg-options string for Bench2Drive data.
    def _build_cfg_options(ann_file: str) -> str:
        if getattr(args, "use_tmpfs_data", False):
            tmpfs_base = "/mnt/bench2drive_ram"
            print("\nUsing tmpfs dataset for baseline/evaluation:")
            if model_type == "VAD":
                print(f"  data_root={tmpfs_base}/bench2drive")
                print(f"  info_root={tmpfs_base}/infos")
                print(f"  map_root={tmpfs_base}/bench2drive/maps")
                print(f"  map_file={tmpfs_base}/b2d_map_infos.pkl")
                return " ".join([
                    f"data_root='{tmpfs_base}/bench2drive'",
                    f"info_root='{tmpfs_base}/infos'",
                    f"map_root='{tmpfs_base}/bench2drive/maps'",
                    f"map_file='{tmpfs_base}/b2d_map_infos.pkl'",
                    f"data.test.ann_file='{ann_file}'",
                ])
            else:  # UniAD
                print(f"  data.test.data_root={tmpfs_base}/bench2drive")
                print(f"  data.test.map_file={tmpfs_base}/infos/b2d_map_infos.pkl")
                return " ".join([
                    f"data.test.data_root='{tmpfs_base}/bench2drive'",
                    f"data.test.map_file='{tmpfs_base}/infos/b2d_map_infos.pkl'",
                    f"data.test.ann_file='{ann_file}'",
                ])
        return f"data.test.ann_file='{ann_file}'"

    # Generate experiment name
    search_algo_params = {
        'num_weights_to_repair': args.repair_num_weights,
        'num_particles': args.repair_num_particles,
        'num_iterations': args.repair_num_iterations,
        'early_stop_patience': args.repair_early_stop_patience,
    }
    exp_name = generate_experiment_name(
        model_name, args.rep_method, args.search_algo, search_algo_params,
        args.fitness, run_idx, base_dir,
        time_horizon=getattr(args, 'time_horizon', 3)
    )
    exp_dir = base_dir / exp_name
    exp_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*80}")
    
    # Step 1: Get baseline JSON
    # If user provides a precomputed baseline JSON, prefer it:
    # - If the file exists: use it directly (regenerate flag is ignored).
    # - If it does NOT exist: generate it into the same folder at the specified path.
    # Otherwise, use a shared "baseline" folder under repo_root to cache/reuse baselines.
    # Note: If baseline JSON exists, it is always reused (regenerate flag is ignored).
    if getattr(args, 'baseline_json', None):
        user_baseline = Path(args.baseline_json)
        if not user_baseline.is_absolute():
            user_baseline = repo_root / user_baseline
        user_baseline = user_baseline.resolve()

        baseline_dir = user_baseline.parent
        baseline_name = user_baseline.name

        if user_baseline.exists():
            print(f"\nUsing provided baseline JSON: {user_baseline}")
            baseline_json = user_baseline
        else:
            print(f"\nRequested baseline JSON does not exist yet: {user_baseline}")
            print("Generating it now into the same folder.")
            baseline_json = generate_baseline_json(
                model_name,
                model_type,
                repair_dataset,
                baseline_dir,
                repo_root,
                args.eval_cuda_device,
                regenerate=True,
                target_filename=baseline_name,
                cfg_options=_build_cfg_options(str(repair_dataset)),
                occ_output_dir=args.occ_output_dir,
            )
    else:
        # Use a shared "baseline" folder under repo_root
        model_lower = model_name.lower()
        dataset_name = Path(repair_dataset).stem
        baseline_dir = repo_root / "baseline"
        baseline_filename = f"{model_lower}_baseline_{dataset_name}.json"
        baseline_json = (baseline_dir / baseline_filename).resolve()

        if baseline_json.exists():
            print(f"\nBaseline JSON already exists in shared baseline folder: {baseline_json}")
            print("Reusing existing baseline (regenerate flag is ignored if file exists).")
        else:
            baseline_json = generate_baseline_json(
                model_name,
                model_type,
                repair_dataset,
                baseline_dir,
                repo_root,
                args.eval_cuda_device,
                regenerate=False,
                target_filename=baseline_filename,
                cfg_options=_build_cfg_options(str(repair_dataset)),
                occ_output_dir=args.occ_output_dir,
            )
    
    if not baseline_json:
        return False
    
    # For UniAD: repair code requires VAD rule format JSON (uses interval-averaged L2, not timestep-specific)
    if model_type == "UniAD":
        vad_rule_json = baseline_json.parent / f"{baseline_json.stem}_vadRule.json"
        if not vad_rule_json.exists():
            print(f"\n{'='*80}")
            print("ERROR: VAD rule format JSON not found for UniAD repair")
            print(f"{'='*80}")
            print(f"Expected file: {vad_rule_json}")
            print(f"Original baseline JSON: {baseline_json}")
            print(f"\nPlease convert the UniAD baseline JSON to VAD rule format first:")
            print(f"  python3 baseline/convert_uniad_to_vad_metrics.py \\")
            print(f"    {baseline_json} \\")
            print(f"    {vad_rule_json}")
            print(f"\nReason: UniAD repair code uses interval-averaged L2 error (VAD rule),")
            print(f"        not timestep-specific L2 error (original UniAD format).")
            print(f"{'='*80}")
            return False
        else:
            print(f"\nUsing VAD rule format JSON for repair: {vad_rule_json}")
        
        # Use VAD rule format JSON for repair
        baseline_json = vad_rule_json
    
    # Step 2: Run repair
    repaired_model = run_repair(
        model_name, model_type, baseline_json, exp_dir, repo_root,
        alpha=args.repair_alpha,
        layers=getattr(args, 'repair_layers', None),
        fitness_type=args.fitness,
        num_particles=args.repair_num_particles,
        num_iterations=args.repair_num_iterations,
        num_weights_to_repair=args.repair_num_weights,
        rep_method=args.rep_method,
        early_stop_patience=args.repair_early_stop_patience,
        search_algo=args.search_algo,
        cuda_device=args.eval_cuda_device,
        time_horizon=getattr(args, 'time_horizon', 3),
        collision_num_workers=getattr(args, 'collision_num_workers', None)
    )
    if not repaired_model:
        return False
    
    # Step 3: Run evaluation
    success = run_evaluation(
        model_name,
        model_type,
        repaired_model,
        eval_dataset,
        exp_dir,
        repo_root,
        cuda_device=args.eval_cuda_device,
        use_tmpfs_data=getattr(args, "use_tmpfs_data", False),
        occ_output_dir=args.eval_occ_output_dir,
    )

    if not success:
        return False

    # Step 4 (optional): Run closed-loop evaluation
    if getattr(args, 'closed_loop_eval', False):
        success = run_closed_loop_evaluation(
            model_name, model_type, repaired_model, exp_dir, repo_root, args
        )

    return success


def main():
    parser = argparse.ArgumentParser(description="Automated experiment runner")
    
    # Model selection
    parser.add_argument('--model-type', choices=['VAD', 'UniAD'], 
                       default='VAD', help='Model type: VAD or UniAD')
    
    # Naming parameters
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name: VAD_base, VAD_tiny, UniAD_base, UniAD_tiny. '
                            'If not specified, defaults based on --model-type (VAD_tiny for VAD, UniAD_tiny for UniAD)')
    parser.add_argument('--rep-method', choices=['Arachne_v1', 'Arachne_v2', 'semSegRep'],
                       default='Arachne_v1',
                       help='Repair method: Arachne_v1 (statistical thresholds, GL+FI on negative only), '
                            'Arachne_v2 (changed/unchanged method with cost_chgd/(1+cost_unchgd)), '
                            'or semSegRep (fixed threshold baseline)')
    parser.add_argument('--search-algo', choices=['PSO', 'DE'], default='PSO',
                       help='Search algorithm: PSO (Particle Swarm Optimization) or DE (Differential Evolution)')
    parser.add_argument('--fitness', choices=['continuous', 'continuous2', 'discrete', 'all'],
                       default='continuous',
                       help='Fitness type: continuous (total L2), continuous2 (total L2 + collision penalty), '
                            'discrete (success rate), or all (run all three types sequentially)')
    parser.add_argument('--run-idx', type=int, default=None,
                       help='Run index (auto-detected if not provided)')
    parser.add_argument('--num-runs', type=int, default=1,
                       help='Number of independent runs')
    
    # Paths
    parser.add_argument('--exp-dir', type=str, default=None,
                       help='Base directory for experiments. If not specified, will auto-generate based on vad-name, rep-method, and search-algo (e.g., ./vad_base_Arachne_v1_PSO_results)')
    parser.add_argument('--repair-dataset', type=str,
                       default='Bench2DriveZoo/data/infos/b2d_infos_repair_tiny.pkl',
                       help='Repair dataset PKL file (Bench2DriveZoo).')
    parser.add_argument('--eval-dataset', type=str,
                       default='Bench2DriveZoo/data/infos/b2d_infos_val_tiny_rest.pkl',
                       help='Evaluation dataset PKL file (Bench2DriveZoo).')
    
    # Repair parameters
    parser.add_argument('--repair-alpha', type=float, default=0.5)
    parser.add_argument('--repair-layers', type=str, default=None,
                       help='Layers to repair (space-separated). '
                            'Default: VAD uses "pts_bbox_head.ego_fut_decoder.0 pts_bbox_head.ego_fut_decoder.2", '
                            'UniAD uses "planning_head.reg_branch.0 planning_head.reg_branch.2"')
    parser.add_argument('--repair-particles-multiplier', type=int, default=2,
                       help='Multiplier for number of PSO particles (particles = multiplier * --repair-num-weights, default: 2)')
    parser.add_argument('--repair-num-iterations', type=int, default=100)
    parser.add_argument('--repair-early-stop-patience', type=int, default=None,
                       help='Early stopping patience for PSO: stop if fitness does not improve for N iterations (default: None, disabled). Example: --repair-early-stop-patience 10')
    parser.add_argument('--repair-num-weights', type=int, default=100)
    parser.add_argument('--collision-num-workers', type=int, default=None,
                       help='Number of worker processes for collision evaluation in repair. '
                            'If omitted, repair script auto-chooses a default.')
    parser.add_argument('--time-horizon', type=int, choices=[1, 2, 3], default=3,
                       help='Time horizon for L2 error and collision: 1s (2 timesteps), 2s (4 timesteps), or 3s (6 timesteps, default). '
                            'This affects both L2 error computation and collision detection.')
    
    # Evaluation parameters
    parser.add_argument('--eval-cuda-device', type=str, default='0')
    parser.add_argument(
        '--use-tmpfs-data',
        type=str2bool,
        default=False,
        help=(
            'If true, assume Bench2Drive data has been copied to tmpfs at '
            '/mnt/bench2drive_ram (see mytools/COPY_TO_TMPFS.sh) and override '
            'data_root/info_root/map_root/map_file to use that path for both '
            'baseline generation and evaluation.'
        ),
    )

    # Closed-loop evaluation (leaderboard)
    # Default False. Enable with `--closed-loop-eval True`.
    parser.add_argument(
        '--closed-loop-eval',
        type=str2bool,
        default=False,
        help='Run closed-loop evaluation after open-loop evaluation (default: False). '
             'Use `--closed-loop-eval True` to enable, `--closed-loop-eval False` to disable.'
    )
    parser.add_argument('--carla-root', type=str, default=None,
                        help='CARLA installation root (needed for closed-loop). If omitted, uses $CARLA_ROOT.')
    parser.add_argument('--closed-loop-routes', type=str, default='leaderboard/data/drivetransformer_bench2drive_dev10.xml',
                        help='Routes XML for closed-loop evaluation')
    parser.add_argument('--closed-loop-team-agent', type=str, default='Bench2DriveZoo/team_code/vad_b2d_agent.py',
                        help='Path to TEAM_AGENT .py for leaderboard evaluation')
    parser.add_argument('--closed-loop-fast', type=str2bool, default=False,
                        help='Use fast version agent (1280x720, no JPEG encode/decode, no 0.8 scaling). '
                             'Use `--closed-loop-fast True` to enable.')
    parser.add_argument('--closed-loop-super-fast', type=str2bool, default=False,
                        help='Use super-fast version agent (640x360, no JPEG encode/decode, no 0.8 scaling). '
                             'Use `--closed-loop-super-fast True` to enable. '
                             'Note: --closed-loop-fast and --closed-loop-super-fast are mutually exclusive.')
    parser.add_argument('--closed-loop-port', type=int, default=30000)
    parser.add_argument('--closed-loop-tm-port', type=int, default=50000)
    parser.add_argument('--closed-loop-gpu-rank', type=int, default=0)
    parser.add_argument('--closed-loop-planner-type', type=str, default='only_traj')
    parser.add_argument('--closed-loop-is-bench2drive', type=str, default='True')
    
    # Baseline JSON control
    parser.add_argument(
        '--baseline-json',
        type=str,
        default=None,
        help='Path to a precomputed baseline JSON file. '
             'If set and the file exists, it will be reused directly (no regeneration). '
             'If it does not exist, it will be generated once at that path.'
    )
    # Baseline generation (only used when --baseline-json is not provided)
    parser.add_argument(
        '--regenerate-baseline',
        type=str2bool,
        default=False,
        help='[DEPRECATED] This flag is ignored if baseline JSON already exists. '
             'Baseline JSON is always reused if found to avoid unnecessary recomputation. '
             'Only used when baseline JSON does not exist yet.'
    )
    parser.add_argument(
        '--occ-output-dir',
        type=str,
        required=True,
        help='Directory to save per-frame occupancy/segmentation for baseline JSON generation.'
    )
    parser.add_argument(
        '--eval-occ-output-dir',
        type=str,
        default=None,
        help='Optional directory to save occupancy during open-loop evaluation.'
    )
    
    args = parser.parse_args()
    
    # Determine model_name from model_type (default if not specified)
    if args.model_name is None:
        if args.model_type == "VAD":
            args.model_name = "VAD_tiny"  # Default VAD model
        elif args.model_type == "UniAD":
            args.model_name = "UniAD_tiny"  # Default UniAD model
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Validate model_name matches model_type
    if args.model_type == "VAD" and not args.model_name.startswith("VAD_"):
        raise ValueError(f"Model name {args.model_name} does not match model type {args.model_type}")
    elif args.model_type == "UniAD" and not args.model_name.startswith("UniAD_"):
        raise ValueError(f"Model name {args.model_name} does not match model type {args.model_type}")
    
    # Calculate repair_num_particles from multiplier and num_weights
    args.repair_num_particles = args.repair_particles_multiplier * args.repair_num_weights

    # Decide occ output dirs (baseline vs eval)
    baseline_occ_output_dir = args.occ_output_dir
    eval_occ_output_dir = args.eval_occ_output_dir
    
    # Auto-generate exp_dir if not specified
    if args.exp_dir is None:
        model_lower = args.model_name.lower()
        rep_method_clean = args.rep_method
        search_algo_clean = args.search_algo
        args.exp_dir = f"./{model_lower}_{rep_method_clean}_{search_algo_clean}_results"
        print(f"Auto-generated exp-dir: {args.exp_dir}")
    
    repo_root = Path(__file__).parent.absolute()
    base_dir = Path(args.exp_dir)
    base_dir.mkdir(parents=True, exist_ok=True)  # 递归创建父目录（如果不存在）

    # If using tmpfs data, sanity-check that the in-memory dataset exists.
    if getattr(args, "use_tmpfs_data", False):
        tmpfs_base = Path("/mnt/bench2drive_ram")
        data_root = tmpfs_base / "bench2drive"
        info_root = tmpfs_base / "infos"
        map_root = tmpfs_base / "bench2drive" / "maps"
        map_file = tmpfs_base / "b2d_map_infos.pkl"

        print("\n[use-tmpfs-data] Enabled. Expecting dataset under /mnt/bench2drive_ram ...")
        missing = []
        if not (data_root.is_dir() and any(data_root.iterdir())):
            missing.append(str(data_root))
        if not (info_root.is_dir() and any(info_root.iterdir())):
            missing.append(str(info_root))
        if not (map_root.is_dir() and any(map_root.iterdir())):
            missing.append(str(map_root))
        if not map_file.is_file():
            missing.append(str(map_file))

        if missing:
            print("ERROR: --use-tmpfs-data is True, but the following paths are missing or empty:")
            for p in missing:
                print(f"  - {p}")
            print("\nPlease first load the Bench2Drive dataset into memory, e.g.:")
            print("  cd", repo_root)
            print("  sudo bash mytools/COPY_TO_TMPFS.sh")
            print("Then re-run this script with --use-tmpfs-data=True.")
            return 1
        else:
            print("Found tmpfs dataset:")
            print(f"  data_root={data_root}")
            print(f"  info_root={info_root}")
            print(f"  map_root={map_root}")
            print(f"  map_file={map_file}")

    # Closed-loop: provide a sensible default CARLA root for this repo
    # (avoids requiring the user to export CARLA_ROOT every time).
    if not args.carla_root:
        candidate = repo_root / "Bench2DriveZoo" / "carla"
        if (candidate / "CarlaUE4.sh").exists() and (candidate / "PythonAPI" / "carla" / "agents").is_dir():
            args.carla_root = str(candidate)
    
    repair_dataset = Path(args.repair_dataset)
    if not repair_dataset.is_absolute():
        repair_dataset = repo_root / repair_dataset
    
    eval_dataset = Path(args.eval_dataset)
    if not eval_dataset.is_absolute():
        eval_dataset = repo_root / eval_dataset
    
    # Run experiments
    # If fitness is 'all', run all three fitness types sequentially
    if args.fitness == 'all':
        fitness_types = ['discrete', 'continuous', 'continuous2']
        total_success_count = 0
        total_experiments = len(fitness_types) * args.num_runs
        
        print(f"\n{'='*80}")
        print(f"Running experiments for ALL fitness types: {fitness_types}")
        print(f"Total experiments: {total_experiments} ({len(fitness_types)} types × {args.num_runs} runs)")
        print(f"{'='*80}")
        
        for fitness_type in fitness_types:
            print(f"\n{'='*80}")
            print(f"FITNESS TYPE: {fitness_type.upper()}")
            print(f"{'='*80}")
            
            # Create a copy of args with modified fitness
            import copy
            fitness_args = copy.deepcopy(args)
            fitness_args.fitness = fitness_type
            
            # Run experiments for this fitness type
            fitness_success_count = 0
            for run_num in range(1, args.num_runs + 1):
                if args.num_runs > 1:
                    print(f"\n{'='*80}")
                    print(f"FITNESS: {fitness_type.upper()} | RUN {run_num}/{args.num_runs}")
                    print(f"{'='*80}")
                
                current_run_idx = args.run_idx if (args.run_idx is not None and run_num == 1) else None
                success = run_single_experiment(
                    current_run_idx, args.model_name, args.model_type, repair_dataset, eval_dataset,
                    base_dir, repo_root, fitness_args
                )
                
                if success:
                    fitness_success_count += 1
                    total_success_count += 1
                    print(f"\n✅ Fitness {fitness_type} | Run {run_num} completed successfully")
                else:
                    print(f"\n❌ Fitness {fitness_type} | Run {run_num} FAILED")
            
            # Summary for this fitness type
            if args.num_runs > 1:
                print(f"\n{'='*80}")
                print(f"FITNESS {fitness_type.upper()} SUMMARY: {fitness_success_count}/{args.num_runs} successful")
                print(f"{'='*80}")
        
        # Overall summary
        print(f"\n{'='*80}")
        print(f"OVERALL SUMMARY: {total_success_count}/{total_experiments} successful")
        print(f"{'='*80}")
        
        return 0 if total_success_count == total_experiments else 1
    else:
        # Normal single fitness type execution
        success_count = 0
        for run_num in range(1, args.num_runs + 1):
            if args.num_runs > 1:
                print(f"\n{'='*80}")
                print(f"RUN {run_num}/{args.num_runs}")
                print(f"{'='*80}")
            
            current_run_idx = args.run_idx if (args.run_idx is not None and run_num == 1) else None
            success = run_single_experiment(
                current_run_idx, args.model_name, args.model_type, repair_dataset, eval_dataset,
                base_dir, repo_root, args
            )
            
            if success:
                success_count += 1
                print(f"\n✅ Run {run_num} completed successfully")
            else:
                print(f"\n❌ Run {run_num} FAILED")
        
        # Summary
        if args.num_runs > 1:
            print(f"\n{'='*80}")
            print(f"SUMMARY: {success_count}/{args.num_runs} successful")
            print(f"{'='*80}")
        
        return 0 if success_count == args.num_runs else 1


if __name__ == "__main__":
    sys.exit(main())
