#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Repair UniAD planning_head.reg_branch with Arachne-style methods.
This is a UniAD-specific variant to avoid touching VAD repair code.
"""

import torch
import torch.nn as nn
import json
import numpy as np
import argparse
import sys
import os
from pathlib import Path

# ---------------------------------------------------------------------
# Bench2DriveZoo integration
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
B2D_ZOO_ROOT = REPO_ROOT / "Bench2DriveZoo"
if str(B2D_ZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(B2D_ZOO_ROOT))

from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model
from mmcv.models.utils.functional import bivariate_gaussian_activation

# Add arachne path relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
arachne_path = os.path.join(REPO_ROOT, 'repair_uniad', 'methods', 'arachne')
if arachne_path not in sys.path:
    sys.path.insert(0, arachne_path)
from arachne_pytorch import ArachnePyTorch, extract_frame_identifiers, build_frame_data_dict

def get_delta_traj(gt_traj):
    """Convert absolute trajectory [T, 2] to delta trajectory [T, 2]."""
    # gt_traj is tensor or numpy
    if not isinstance(gt_traj, torch.Tensor):
        gt_traj = torch.tensor(gt_traj, dtype=torch.float32)
    
    delta_traj = torch.zeros_like(gt_traj)
    # First point: delta from (0,0) is just the point itself
    delta_traj[0] = gt_traj[0]
    # Subsequent points: difference from previous point
    delta_traj[1:] = gt_traj[1:] - gt_traj[:-1]
    return delta_traj



class MultiLayerWrapper(nn.Module):
    """
    Wrapper for UniAD reg_branch layers.
    Stores target layers as separate attributes for localization,
    but uses full decoder for evaluation.
    """
    def __init__(self, target_layers_dict, full_decoder):
        super().__init__()
        for i, (name, layer) in enumerate(target_layers_dict.items()):
            setattr(self, f'layer{i}', layer)
        self.full_decoder = full_decoder
        self.num_layers = len(target_layers_dict)
        self.layer_names = [f'layer{i}' for i in range(len(target_layers_dict))]
        self.target_layer_names = list(target_layers_dict.keys())
        self.target_indices = {}
        for name in self.target_layer_names:
            for i, layer in enumerate(full_decoder):
                if hasattr(layer, '__class__') and layer.__class__.__name__ == 'Linear':
                    if i not in self.target_indices.values():
                        self.target_indices[name] = i
                        break

    def forward(self, x):
        """
        Forward pass through reg_branch.
        
        IMPORTANT: reg_branch is a nn.Sequential that already contains ReLU activations
        between layers. We should NOT add additional ReLU activations here, as that would
        change the output even when perturbation=0.
        
        The reg_branch structure is:
        nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),  # Already in Sequential
            nn.Linear(embed_dims, planning_steps * 2),
        )
        
        So we just need to call the layers in order, using repaired layers for target layers.
        """
        # If full_decoder is a Sequential, we need to iterate through its modules
        # and apply them in order, replacing target layers with repaired versions
        if isinstance(self.full_decoder, nn.Sequential):
            # For Sequential, we iterate through the modules
            for module_idx, module in enumerate(self.full_decoder):
                layer_to_use = None
                # Check if this module index corresponds to a target layer
                for i, target_name in enumerate(self.target_layer_names):
                    target_decoder_idx = self.target_indices[target_name]
                    if module_idx == target_decoder_idx:
                        layer_to_use = getattr(self, f'layer{i}')
                        break
                if layer_to_use is None:
                    layer_to_use = module
                x = layer_to_use(x)
        else:
            # Fallback: original logic for non-Sequential decoders
            for decoder_idx, layer in enumerate(self.full_decoder):
                layer_to_use = None
                for i, target_name in enumerate(self.target_layer_names):
                    target_decoder_idx = self.target_indices[target_name]
                    if decoder_idx == target_decoder_idx:
                        layer_to_use = getattr(self, f'layer{i}')
                        break
                if layer_to_use is None:
                    layer_to_use = self.full_decoder[decoder_idx]
                x = layer_to_use(x)
        return x


def load_uniad_model(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    if hasattr(cfg, "model"):
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        
        # Fix relative paths in config (e.g., anchor_info_path)
        # Config paths are relative to Bench2DriveZoo, convert to relative to repo root
        if hasattr(cfg.model, 'motion_head') and cfg.model.motion_head is not None:
            if isinstance(cfg.model.motion_head, dict):
                anchor_path = cfg.model.motion_head.get('anchor_info_path', None)
                if anchor_path and not os.path.isabs(anchor_path):
                    # Convert to path relative to repo root: Bench2DriveZoo/data/others/...
                    # This matches the pattern used in team_code/uniad_b2d_agent.py
                    repo_relative_path = os.path.join('Bench2DriveZoo', anchor_path)
                    # Verify the file exists
                    full_path = REPO_ROOT / repo_relative_path
                    if full_path.exists():
                        cfg.model.motion_head['anchor_info_path'] = repo_relative_path
                    else:
                        print(f"WARNING: anchor_info_path not found: {full_path}")

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    map_location = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'
    _ = load_checkpoint(model, checkpoint_path, map_location=map_location)
    model.eval()
    return model, cfg


def predict_traj_from_features(model, ego_features):
    """Run UniAD reg_branch using plan_query features to get absolute trajectory."""
    planning_head = model.planning_head
    planning_steps = int(getattr(planning_head, 'planning_steps', 6))
    x = ego_features.unsqueeze(0)  # [1, embed_dims]
    out = planning_head.reg_branch(x)  # [1, planning_steps*2]
    out = out.view(-1, planning_steps, 2)  # [1, steps, 2]
    out = torch.cumsum(out, dim=1)
    out[0] = bivariate_gaussian_activation(out[0])
    return out[0]  # [steps, 2]


def extract_features(model, data_infos, threshold_good, threshold_bad,
                     mean_l2=None, std_l2=None, alpha=0.5, rep_method='Arachne_v1', time_horizon=3):
    model.eval()
    pos_feat, neg_feat = [], []
    pos_gt, neg_gt = [], []
    pos_l2, neg_l2 = [], []
    collision_l2 = []
    negative_no_collision_l2 = []
    collision_count = 0
    positive_no_collision_count = 0
    middle_no_collision_count = 0
    negative_no_collision_count = 0
    processed_frames = 0

    if rep_method == 'Arachne_v1':
        if mean_l2 is None or std_l2 is None:
            raise ValueError("Arachne_v1 classification requires mean_l2 and std_l2 statistics.")
        collision_threshold_good = mean_l2 - alpha * std_l2
        collision_threshold_bad = mean_l2 + alpha * std_l2
    else:
        collision_threshold_good = threshold_good
        collision_threshold_bad = threshold_bad

    timesteps_for_horizon = {1: 2, 2: 4, 3: 6}
    num_timesteps = timesteps_for_horizon.get(time_horizon, 6)

    with torch.no_grad():
        for info in data_infos:
            if not info.get('fut_valid_flag', False):
                continue

            ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
            processed_frames += 1
            pred_traj = predict_traj_from_features(model, ego_features)
            gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)
            min_len = min(num_timesteps, int(pred_traj.shape[0]), int(gt_traj.shape[0]))
            l2_error = torch.norm(pred_traj[:min_len] - gt_traj[:min_len], dim=1).mean().item()

            if rep_method == 'semSegRep':
                if l2_error < threshold_good:
                    pos_feat.append(ego_features)
                    pos_gt.append(get_delta_traj(gt_traj))
                    pos_l2.append(l2_error)
                    positive_no_collision_count += 1
                else:
                    neg_feat.append(ego_features)
                    neg_gt.append(get_delta_traj(gt_traj))
                    neg_l2.append(l2_error)
                    negative_no_collision_l2.append(l2_error)
                    negative_no_collision_count += 1
            else:
                collision_field = f'plan_obj_box_col_{time_horizon}s'
                col_value = info.get(collision_field, 0.0)
                has_collision = (col_value > 0)
                if has_collision:
                    neg_feat.append(ego_features)
                    neg_gt.append(get_delta_traj(gt_traj))
                    neg_l2.append(l2_error)
                    collision_l2.append(l2_error)
                    collision_count += 1
                else:
                    if l2_error < collision_threshold_good:
                        pos_feat.append(ego_features)
                        pos_gt.append(get_delta_traj(gt_traj))
                        pos_l2.append(l2_error)
                        positive_no_collision_count += 1
                    elif l2_error > collision_threshold_bad:
                        neg_feat.append(ego_features)
                        neg_gt.append(get_delta_traj(gt_traj))
                        neg_l2.append(l2_error)
                        negative_no_collision_l2.append(l2_error)
                        negative_no_collision_count += 1
                    else:
                        middle_no_collision_count += 1

    return pos_feat, neg_feat, pos_gt, neg_gt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=str(B2D_ZOO_ROOT / 'adzoo/uniad/configs/stage2_e2e/base_e2e_b2d.py'),
        help='UniAD config path (Bench2DriveZoo).'
    )
    parser.add_argument(
        '--checkpoint',
        default=str(B2D_ZOO_ROOT / 'ckpts/uniad_base_b2d.pth'),
        help='UniAD checkpoint path (download into Bench2DriveZoo/ckpts).'
    )
    parser.add_argument(
        '--json',
        default=str(REPO_ROOT / 'baseline/uniad_base_b2d_infos_val_partA_25clips.json'),
        help='Collected per-frame JSON for repair (generated by adzoo/uniad/test.py --collect-data).'
    )
    parser.add_argument('--layers', nargs='+',
                       default=['planning_head.reg_branch.0'],
                       help='List of layer names to repair (space-separated). '
                            'Default: planning_head.reg_branch.0 (256x256 hidden layer). '
                            'Can also include planning_head.reg_branch.2 (256x12 output layer).')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--rep-method', choices=['Arachne_v1', 'Arachne_v2', 'semSegRep'],
                       default='Arachne_v1')
    parser.add_argument('--search-algo', choices=['PSO', 'DE'], default='PSO')
    parser.add_argument('--num-particles', type=int, default=100)
    parser.add_argument('--num-iterations', type=int, default=20)
    parser.add_argument('--early-stop-patience', type=int, default=5)
    parser.add_argument('--output-dir', default='./repair_uniad_planning_head')
    parser.add_argument('--num-weights-to-repair', type=int, default=None)
    parser.add_argument('--fitness-type', type=str, default='discrete',
                       choices=['discrete', 'continuous', 'continuous2'])
    parser.add_argument('--time-horizon', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--collision-num-workers', type=int, default=None,
                       help='Number of worker processes for collision evaluation. '
                            'If omitted, auto-choose a safe default.')
    parser.add_argument(
        '--use-cached-eval',
        action='store_true',
        help='Use cached JSON/occ evaluation for fitness (default: on).'
    )
    args = parser.parse_args()
    # Default to True (cached eval is on by default)
    # If user doesn't specify --use-cached-eval, it will be False from store_true,
    # so we set it to True as default
    if not args.use_cached_eval:
        args.use_cached_eval = True

    if args.rep_method in ['Arachne_v1', 'Arachne_v2'] and args.alpha <= 0:
        parser.error(f"--alpha must be > 0 for {args.rep_method}, current value: {args.alpha}")

    print("="*70)
    print(f"Repair Method: {args.rep_method}")
    print(f"Repair UniAD layers: {', '.join(args.layers)}")
    print("="*70)

    print("\n[1/4] Loading UniAD model...")
    uniad_model, cfg = load_uniad_model(args.config, args.checkpoint)

    print("\nTarget layers:")
    target_layers = {}
    total_params = 0
    for layer_name in args.layers:
        try:
            layer = uniad_model.get_submodule(layer_name)
            if not isinstance(layer, nn.Linear):
                print(f"  Error: Layer '{layer_name}' is not Linear!")
                continue
            target_layers[layer_name] = layer
            layer_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            total_params += layer_params
            
            # Print weight statistics
            weight_np = layer.weight.detach().cpu().numpy()
            weight_min = float(np.min(weight_np))
            weight_max = float(np.max(weight_np))
            weight_range = weight_max - weight_min
            weight_mean = float(np.mean(weight_np))
            weight_std = float(np.std(weight_np))
            
            print(f"  {layer_name}: {layer.weight.shape} ({layer_params} params)")
            print(f"    Weight statistics:")
            print(f"      Min: {weight_min:.6f}")
            print(f"      Max: {weight_max:.6f}")
            print(f"      Range: {weight_range:.6f} (Max - Min)")
            print(f"      Mean: {weight_mean:.6f}")
            print(f"      Std: {weight_std:.6f}")
            print(f"      Search bounds: [{weight_min - weight_range * 1.0:.6f}, {weight_max + weight_range * 1.0:.6f}] (extended by ±100%)")
            print(f"      Initialization perturbation: ±{weight_range * 0.05:.6f} (5% of range)")
        except AttributeError:
            print(f"  Error: Layer '{layer_name}' not found!")

    print(f"  Total: {total_params} parameters across {len(args.layers)} layer(s)")

    print("\n[2/4] Loading data...")
    with open(args.json, 'r') as f:
        data_infos = json.load(f)

    l2_field = f'plan_L2_{args.time_horizon}s'
    total_valid_frames_in_json = sum(
        1 for info in data_infos if info.get('fut_valid_flag', False) and info.get(l2_field) is not None
    )
    print(f"  Valid frames in JSON: {total_valid_frames_in_json}")

    if args.rep_method in ['Arachne_v1', 'Arachne_v2']:
        print("\nComputing L2 error statistics to automatically determine thresholds...")
        uniad_model.eval()
        l2_errors = []
        timesteps_for_horizon = {1: 2, 2: 4, 3: 6}
        num_timesteps = timesteps_for_horizon.get(args.time_horizon, 6)
        with torch.no_grad():
            for info in data_infos:
                if not info.get('fut_valid_flag', False):
                    continue
                ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
                pred_traj = predict_traj_from_features(uniad_model, ego_features)
                gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)
                min_len = min(num_timesteps, int(pred_traj.shape[0]), int(gt_traj.shape[0]))
                l2_error = torch.norm(pred_traj[:min_len] - gt_traj[:min_len], dim=1).mean().item()
                if not np.isnan(l2_error) and not np.isinf(l2_error):
                    l2_errors.append(float(l2_error))
        if len(l2_errors) == 0:
            parser.error("Error: No valid L2 error data found! Cannot automatically calculate thresholds.")
        l2_errors = np.array(l2_errors)
        mean_l2 = float(np.mean(l2_errors))
        std_l2 = float(np.std(l2_errors))
        threshold_good = mean_l2 - args.alpha * std_l2
        threshold_bad = mean_l2 + args.alpha * std_l2
        mean_l2_for_classification = mean_l2
        std_l2_for_classification = std_l2
    else:
        mean_l2_for_classification = None
        std_l2_for_classification = None
        print("\nUsing automatic median threshold for semSegRep (--alpha parameter is ignored)...")
        uniad_model.eval()
        l2_errors = []
        timesteps_for_horizon = {1: 2, 2: 4, 3: 6}
        num_timesteps = timesteps_for_horizon.get(args.time_horizon, 6)
        with torch.no_grad():
            for info in data_infos:
                if not info.get('fut_valid_flag', False):
                    continue
                ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
                pred_traj = predict_traj_from_features(uniad_model, ego_features)
                gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)
                min_len = min(num_timesteps, int(pred_traj.shape[0]), int(gt_traj.shape[0]))
                l2_error = torch.norm(pred_traj[:min_len] - gt_traj[:min_len], dim=1).mean().item()
                if not np.isnan(l2_error) and not np.isinf(l2_error):
                    l2_errors.append(float(l2_error))
        if len(l2_errors) == 0:
            parser.error("Error: No valid L2 error data found! Cannot calculate median threshold.")
        l2_errors = np.array(l2_errors)
        threshold_good = float(np.median(l2_errors))
        threshold_bad = float(np.median(l2_errors))

    pos_feat, neg_feat, pos_gt, neg_gt = extract_features(
        uniad_model, data_infos,
        threshold_good=threshold_good,
        threshold_bad=threshold_bad,
        mean_l2=mean_l2_for_classification,
        std_l2=std_l2_for_classification,
        alpha=args.alpha,
        rep_method=args.rep_method,
        time_horizon=args.time_horizon
    )

    if len(pos_feat) == 0 or len(neg_feat) == 0:
        print("No positive/negative features found. Aborting repair.")
        return

    # Wrap target layers for Arachne
    reg_branch = uniad_model.planning_head.reg_branch
    wrapper = MultiLayerWrapper(target_layers, reg_branch)

    # Stack features and create dummy labels (arachne_pytorch expects tuple (features, labels))
    if pos_feat:
        pos_feat_dim = pos_feat[0].shape[0]
        # Pass actual ground truth labels instead of empty placeholders
        # Note: gt_traj shape is [6, 2], so we flatten it or keep as is depending on arachne expectation.
        # Arachne expects labels. For regression, labels are the GT trajectories.
        input_pos = (torch.stack(pos_feat), torch.stack(pos_gt))
    else:
        pos_feat_dim = len(data_infos[0].get('ego_features', [])) if data_infos else 512
        input_pos = (torch.empty(0, pos_feat_dim), torch.empty(0, 1))
    if neg_feat:
        neg_feat_dim = neg_feat[0].shape[0]
        # Pass actual ground truth labels instead of empty placeholders
        input_neg = (torch.stack(neg_feat), torch.stack(neg_gt))
    else:
        neg_feat_dim = len(data_infos[0].get('ego_features', [])) if data_infos else 512
        input_neg = (torch.empty(0, neg_feat_dim), torch.empty(0, 1))

    # Build Arachne
    arachne = ArachnePyTorch()
    if args.collision_num_workers is None:
        print("[WARN] --collision-num-workers not set; defaulting to 1 (serial). "
              "This may be slow. Consider setting it explicitly, e.g. --collision-num-workers 8.")
        args.collision_num_workers = 1
    if args.collision_num_workers < 1:
        parser.error("--collision-num-workers must be >= 1")
    
    arachne.set_options(
        num_particles=args.num_particles,
        num_iterations=args.num_iterations,
        num_weights_to_repair=args.num_weights_to_repair,
        target_layers=wrapper.layer_names,
        early_stop_patience=args.early_stop_patience,
        optimization_algorithm=args.search_algo,
        pred_traj_is_delta=True,
        apply_bivariate_activation=True,
        collision_num_workers=args.collision_num_workers,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Localize
    use_changed_unchanged = (args.rep_method == 'Arachne_v2')
    if use_changed_unchanged:
        weights = arachne.localize(
            wrapper, input_neg, input_pos=input_pos,
            output_dir=output_dir, use_changed_unchanged=True
        )
    else:
        weights = arachne.localize(wrapper, input_neg, output_dir=output_dir)

    # Build frame data for cached open-loop evaluation
    frame_data_dict = None
    positive_frames = None
    negative_frames = None
    if args.use_cached_eval:
        frame_data_dict = build_frame_data_dict(args.json, frame_identifiers=None)
        if args.rep_method not in ['Arachne_v1', 'Arachne_v2']:
            positive_frames, negative_frames = extract_frame_identifiers(
                args.json,
                threshold_good=threshold_good,
                threshold_bad=threshold_bad
            )

    # Optimize
    repaired_wrapper, fitness_history = arachne.optimize(
        wrapper, weights, input_neg, input_pos, output_dir, verbose=1,
        use_cached_eval=args.use_cached_eval,
        frame_data_dict=frame_data_dict if args.use_cached_eval else None,
        positive_frames=positive_frames if args.use_cached_eval else None,
        negative_frames=negative_frames if args.use_cached_eval else None,
        threshold_good=threshold_good,
        threshold_bad=threshold_bad,
        fitness_type=args.fitness_type,
        rep_method=args.rep_method,
        time_horizon=getattr(args, 'time_horizon', 3)
    )

    # Apply repaired weights back to full UniAD model
    print("\n" + "="*70)
    print("Copying repaired weights to UniAD model")
    print("="*70)
    
    with torch.no_grad():
        for i, (orig_name, orig_layer) in enumerate(target_layers.items()):
            wrapper_layer = getattr(repaired_wrapper, f'layer{i}')
            orig_layer.weight.copy_(wrapper_layer.weight)
            if orig_layer.bias is not None and wrapper_layer.bias is not None:
                orig_layer.bias.copy_(wrapper_layer.bias)
            print(f"  Copied {orig_name}")
    
    # Verify: ensure all non-target layers are identical to original
    print("\nVerifying model integrity before saving...")
    original_checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if isinstance(original_checkpoint, dict) and 'state_dict' in original_checkpoint:
        original_state = original_checkpoint['state_dict']
    else:
        original_state = original_checkpoint
    
    current_state = uniad_model.state_dict()
    # Move current_state to CPU for comparison (original_state is on CPU)
    current_state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in current_state.items()}
    
    # Build target_layer_keys dynamically from args.layers
    target_layer_keys = []
    for layer_name in args.layers:
        target_layer_keys.append(layer_name + '.weight')
        target_layer_keys.append(layer_name + '.bias')
    
    non_target_keys = [k for k in original_state.keys() if k not in target_layer_keys]
    mismatched = []
    for key in non_target_keys:
        if key not in current_state_cpu:
            mismatched.append((key, "missing"))
        elif not torch.equal(original_state[key], current_state_cpu[key]):
            mismatched.append((key, "different"))
    
    if mismatched:
        print(f"  WARNING: Found {len(mismatched)} mismatched keys!")
        for key, reason in mismatched[:10]:
            print(f"    - {key} ({reason})")
        if len(mismatched) > 10:
            print(f"    ... and {len(mismatched) - 10} more")
        raise RuntimeError("Model integrity check failed! Non-target layers were modified.")
    else:
        print(f"  Verified: All {len(non_target_keys)} non-target layer keys are identical to original")
    
    # Save: preserve original checkpoint structure (including optimizer, epoch, etc.)
    save_path = output_dir / 'repaired_model.pth'
    if isinstance(original_checkpoint, dict) and 'state_dict' in original_checkpoint:
        # Original checkpoint has metadata structure
        repaired_checkpoint = original_checkpoint.copy()
        repaired_checkpoint['state_dict'] = uniad_model.state_dict()
        torch.save(repaired_checkpoint, save_path)
        print(f"\nSaved repaired model to: {save_path}")
        print(f"  Preserved original checkpoint structure with metadata")
        print(f"  Total keys in state_dict: {len(current_state)}")
        print(f"  Modified keys: {len(target_layer_keys)}")
        print(f"  Unchanged keys: {len(non_target_keys)}")
    else:
        # Original checkpoint is just state_dict
        torch.save(uniad_model.state_dict(), save_path)
        print(f"\nSaved repaired model to: {save_path}")
        print(f"  Total keys: {len(current_state)}")
        print(f"  Modified keys: {len(target_layer_keys)}")
        print(f"  Unchanged keys: {len(non_target_keys)}")

    print(f"\nRepair finished. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
