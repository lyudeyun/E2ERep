#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simultaneously repair ego_fut_decoder.0 and ego_fut_decoder.2
"""

import torch
import torch.nn as nn
import json
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# ---------------------------------------------------------------------
# Bench2DriveZoo integration
# ---------------------------------------------------------------------
# This repo vendors a custom `mmcv/` inside `Bench2DriveZoo/`. To make sure we
# build the B2D VAD model (and not the pip-installed OpenMMLab mmcv), we must
# put Bench2DriveZoo on sys.path before importing mmcv.
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
B2D_ZOO_ROOT = REPO_ROOT / "Bench2DriveZoo"
if str(B2D_ZOO_ROOT) not in sys.path:
    sys.path.insert(0, str(B2D_ZOO_ROOT))

# Build model using Bench2DriveZoo's registry/builders
from mmcv.utils import Config, load_checkpoint
from mmcv.models import build_model
# Add arachne path relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
arachne_path = os.path.join(script_dir, 'methods', 'arachne')
if arachne_path not in sys.path:
    sys.path.insert(0, arachne_path)
from arachne_pytorch import ArachnePyTorch, extract_frame_identifiers, build_frame_data_dict


class MultiLayerWrapper(nn.Module):
    """
    Wrapper for VAD decoder layers.
    Stores target layers as separate attributes for localization,
    but uses full decoder for evaluation.
    """
    def __init__(self, target_layers_dict, full_decoder):
        super().__init__()
        # Store target layers separately (for localization/gradient computation)
        for i, (name, layer) in enumerate(target_layers_dict.items()):
            setattr(self, f'layer{i}', layer)
        
        # Store the complete decoder (for evaluation)
        self.full_decoder = full_decoder
        
        self.num_layers = len(target_layers_dict)
        self.layer_names = [f'layer{i}' for i in range(len(target_layers_dict))]
        self.target_layer_names = list(target_layers_dict.keys())
        
        # Map target layer names to indices in full_decoder
        self.target_indices = {}
        for name in self.target_layer_names:
            for i, layer in enumerate(full_decoder):
                if hasattr(layer, '__class__') and layer.__class__.__name__ == 'Linear':
                    if i not in self.target_indices.values():
                        self.target_indices[name] = i
                        break
    
    def forward(self, x):
        # Always pass through full decoder (for both training and eval)
        # Only the target layers' weights will be modified during PSO
        for decoder_idx, layer in enumerate(self.full_decoder):
            # Check if this is one of our target layers
            layer_to_use = None
            for i, target_name in enumerate(self.target_layer_names):
                target_decoder_idx = self.target_indices[target_name]
                if decoder_idx == target_decoder_idx:
                    # Use our modified layer
                    layer_to_use = getattr(self, f'layer{i}')
                    break
            
            # If not a target layer, use original from full_decoder
            if layer_to_use is None:
                layer_to_use = self.full_decoder[decoder_idx]
            
            x = layer_to_use(x)
            
            # Add ReLU between layers (matching the decoder structure)
            if decoder_idx + 1 < len(self.full_decoder):
                x = torch.relu(x)
        
        return x  # Full trajectory output [batch, 36]


def load_vad_model(config_path, checkpoint_path):
    """Load VAD model from config and checkpoint."""
    cfg = Config.fromfile(config_path)
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # Match adzoo/vad/test.py behavior
    if hasattr(cfg, "model"):
        cfg.model.pretrained = None
        cfg.model.train_cfg = None

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load to GPU if available for speed; fall back to CPU
    if torch.cuda.is_available():
        map_location = f'cuda:{torch.cuda.current_device()}'
    else:
        map_location = 'cpu'
    _ = load_checkpoint(model, checkpoint_path, map_location=map_location)
    model.eval()
    
    return model, cfg


def extract_features(model, data_infos, threshold_good, threshold_bad, 
                     mean_l2=None, std_l2=None, alpha=0.5, rep_method='Arachne_v1', time_horizon=3):
    """
    Extract frame features and compute L2 errors for localization-time classification.
    
    Classification logic depends on rep_method:
    
    For semSegRep:
        - L2 < threshold_good: positive (counted in positive_no_collision_count / positive_count)
        - L2 >= threshold_good: negative (counted in negative_no_collision_count / negative_count)
        - Collision flags are ignored for classification, but still recorded for downstream statistics
    
        For Arachne_v1:
        1. If collision == True: frame = negative (collision_count)
        2. Else:
           - If L2_error < mean - alpha * std: frame = positive (positive_no_collision_count)
           - If L2_error > mean + alpha * std: frame = negative (negative_no_collision_count)
           - Else: frame = middle/neutral (middle_no_collision_count) and excluded from localization data
        Aggregated counts:
           - positive_count = positive_no_collision_count
           - negative_count = collision_count + negative_no_collision_count
    
    Parameters
    ----------
    model : nn.Module
        VAD model
    data_infos : list
        List of frame data dictionaries
    threshold_good : float
        Positive threshold. For semSegRep: fixed threshold (threshold_good == threshold_bad).
        For Arachne_v1: lower threshold derived from mean - alpha * std when statistics are enabled.
    threshold_bad : float
        Negative threshold. For semSegRep: same as threshold_good (not used separately).
        For Arachne_v1: upper threshold derived from mean + alpha * std when statistics are enabled.
    mean_l2 : float, optional (required for Arachne_v1)
        Mean L2 error for statistical thresholds (Arachne_v1 only)
    std_l2 : float, optional (required for Arachne_v1)
        Std L2 error for statistical thresholds (Arachne_v1 only)
    alpha : float
        For Arachne_v1: Alpha coefficient for statistical threshold calculation (default: 0.5).
        For semSegRep: always uses median L2 threshold (alpha parameter is ignored).
    rep_method : str
        Repair method: 'Arachne_v1' or 'semSegRep'
    """
    model.eval()
    
    pos_feat = []
    neg_feat = []
    pos_l2 = []
    neg_l2 = []
    collision_l2 = []              # Track L2 values for collision-based negatives separately
    negative_no_collision_l2 = []  # Track L2 values for non-collision negatives separately
    collision_count = 0
    positive_no_collision_count = 0
    middle_no_collision_count = 0
    negative_no_collision_count = 0
    processed_frames = 0
    
    # Determine classification thresholds
    if rep_method == 'Arachne_v1':
        if mean_l2 is None or std_l2 is None:
            raise ValueError("Arachne_v1 classification requires mean_l2 and std_l2 statistics.")
        collision_threshold_good = mean_l2 - alpha * std_l2
        collision_threshold_bad = mean_l2 + alpha * std_l2
        print(f"  Using statistical thresholds for classification:")
        print(f"    collision_threshold_good = {mean_l2:.6f} - {alpha} * {std_l2:.6f} = {collision_threshold_good:.6f}")
        print(f"    collision_threshold_bad = {mean_l2:.6f} + {alpha} * {std_l2:.6f} = {collision_threshold_bad:.6f}")
        print(f"    (Note: These may differ from threshold_good/bad if computed from different data)")
    else:
        # semSegRep: fixed threshold baseline
        collision_threshold_good = threshold_good
        collision_threshold_bad = threshold_bad
        print(f"  Using fixed threshold for classification:")
        print(f"    threshold_good = {threshold_good:.6f}")
        print(f"    threshold_bad = {threshold_bad:.6f}")
    
    with torch.no_grad():
        for info in data_infos:
            # Skip invalid frames (must have fut_valid_flag=True, same as build_frame_data_dict)
            if not info.get('fut_valid_flag', False):
                continue
            
            # Load ego_features
            ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
            processed_frames += 1
            
            # Forward through decoder to get prediction
            pred_traj = model.pts_bbox_head.ego_fut_decoder(ego_features.unsqueeze(0))
            
            # Compute L2 error
            gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)  # [T, 2]
            cmd_idx = info.get('ego_fut_cmd_idx', 0)
            
            # Reshape prediction based on model attributes.
            # For Bench2Drive VAD, ego_fut_mode is typically 6 (6 high-level commands),
            # and valid_fut_ts is typically 6 (future steps).
            ego_fut_mode = int(getattr(model.pts_bbox_head, 'ego_fut_mode', 6))
            fut_ts = int(getattr(model.pts_bbox_head, 'valid_fut_ts', 6))
            pred_traj = pred_traj.view(ego_fut_mode, fut_ts, 2)
            cmd_idx = int(np.clip(cmd_idx, 0, ego_fut_mode - 1))
            pred_mode = pred_traj[cmd_idx]  # [fut_ts, 2] (DELTA)
            
            # IMPORTANT: VAD decoder outputs delta (displacement) values, not absolute positions
            # Convert delta to absolute positions using cumsum before computing L2 error
            pred_mode_absolute = torch.cumsum(pred_mode, dim=0)  # [fut_ts, 2]
            
            # Compute L2 error for the specified time horizon
            # Each timestep = 0.5s, so:
            # 1s = 2 timesteps (2 trajectory points)
            # 2s = 4 timesteps (4 trajectory points)
            # 3s = 6 timesteps (6 trajectory points, full trajectory)
            # Dictionary mapping: {time_horizon_in_seconds: number_of_timesteps}
            timesteps_for_horizon = {1: 2, 2: 4, 3: 6}  # {1秒: 2步, 2秒: 4步, 3秒: 6步}
            num_timesteps = timesteps_for_horizon.get(time_horizon, 6)
            min_len = min(num_timesteps, int(pred_mode_absolute.shape[0]), int(gt_traj.shape[0]))
            l2_error = torch.norm(pred_mode_absolute[:min_len] - gt_traj[:min_len], dim=1).mean().item()
            
            # Classification logic
            # For semSegRep: simple fixed threshold, no collision check, no neutral category
            if rep_method == 'semSegRep':
                # Fixed threshold: L2 < threshold = positive, L2 >= threshold = negative
                # No middle/neutral category for semSegRep
                if l2_error < threshold_good:  # threshold_good == threshold_bad for semSegRep
                    pos_feat.append(ego_features)
                    pos_l2.append(l2_error)
                    positive_no_collision_count += 1
                else:  # l2_error >= threshold_good
                    neg_feat.append(ego_features)
                    neg_l2.append(l2_error)
                    negative_no_collision_l2.append(l2_error)  # Track for consistency
                    negative_no_collision_count += 1
            else:
                # For Arachne_v1: classification logic
                # 1. If collision == True: frame = negative
                # 2. Else:
                #    - If L2_error < mean - 0.5*std: frame = positive
                #    - If L2_error > mean + 0.5*std: frame = negative
                #    - Else: frame = neutral
                
                # Check for collision using the same time horizon as L2 error
                collision_field = f'plan_obj_box_col_{time_horizon}s'
                col_value = info.get(collision_field, 0.0)
                has_collision = (col_value > 0)
                
                if has_collision:
                    # Collision frames are always negative (strong system-level metric)
                    neg_feat.append(ego_features)
                    neg_l2.append(l2_error)
                    collision_l2.append(l2_error)
                    collision_count += 1
                else:
                    # No collision: classify based on L2 error using statistical thresholds
                    if l2_error < collision_threshold_good:  # L2 < mean - alpha*std
                        pos_feat.append(ego_features)
                        pos_l2.append(l2_error)
                        positive_no_collision_count += 1
                    elif l2_error > collision_threshold_bad:  # L2 > mean + alpha*std
                        neg_feat.append(ego_features)
                        neg_l2.append(l2_error)
                        negative_no_collision_l2.append(l2_error)
                        negative_no_collision_count += 1
                    else:  # mean - alpha*std <= L2 <= mean + alpha*std
                        middle_no_collision_count += 1
    
    # Print classification results
    # Fall back to len(pos_feat) in case positive_no_collision_count wasn't updated (e.g., legacy paths)
    if positive_no_collision_count == 0 and len(pos_feat) > 0:
        positive_no_collision_count = len(pos_feat)
    # Aggregate counts for compatibility with downstream logic
    positive_count = positive_no_collision_count
    negative_count = collision_count + negative_no_collision_count
    total_frames_classified = positive_count + negative_count + middle_no_collision_count
    
    if rep_method in ['Arachne_v1', 'Arachne_v2']:
        # 4 categories for Arachne_v1 and Arachne_v2 (same classification logic)
        method_name = 'Arachne_v1' if rep_method == 'Arachne_v1' else 'Arachne_v2'
        print(f"\nData classification ({method_name} logic - 4 categories, time_horizon={time_horizon}s):")
        print(f"  Category 1 - Positive (no collision, L2 < mean - 0.5*std): {positive_no_collision_count}")
        print(f"  Category 2 - Negative (collision): {collision_count}")
        print(f"  Category 3 - Negative (no collision, L2 > mean + 0.5*std): {negative_no_collision_count}")
        print(f"  Category 4 - Middle (no collision, mean - 0.5*std <= L2 <= mean + 0.5*std): {middle_no_collision_count}")
        print(f"  Total positive: {positive_count}")
        print(f"  Total negative (Category 2 + 3): {negative_count}")
        print(f"  Total frames: {total_frames_classified}")
        print(f"  Processed frames: {processed_frames}")
        
        if total_frames_classified != processed_frames:
            raise ValueError(
                f"{method_name} classification mismatch: "
                f"processed {processed_frames} valid frames but classified {total_frames_classified} "
                f"(positive={positive_no_collision_count}, collision={collision_count}, "
                f"negative_no_collision={negative_no_collision_count}, middle={middle_no_collision_count})."
            )
    else:
        # 2 categories for semSegRep (simple fixed threshold, no neutral)
        print(f"\nData classification (semSegRep logic - 2 categories, time_horizon={time_horizon}s, fixed threshold={threshold_good:.6f}):")
        print(f"  Positive (L2 < {threshold_good:.6f}): {len(pos_feat)}")
        print(f"  Negative (L2 >= {threshold_good:.6f}): {len(neg_feat)}")
        print(f"  Total frames: {len(pos_feat) + len(neg_feat)}")
        positive_count = len(pos_feat)
        negative_count = len(neg_feat)
        
        if (positive_count + negative_count) != processed_frames:
            raise ValueError(
                "semSegRep classification mismatch: "
                f"processed {processed_frames} valid frames but classified {positive_count + negative_count} "
                f"(positive={positive_count}, negative={negative_count})."
            )
    
    if collision_count > 0 and collision_l2:
        print(f"\n  Category 2 details (collision negatives):")
        print(f"    L2 range: [{min(collision_l2):.3f}, {max(collision_l2):.3f}]")
    if negative_no_collision_l2:
        threshold_display = collision_threshold_bad if rep_method in ['Arachne_v1', 'Arachne_v2'] else threshold_bad
        print(f"\n  Category 3 details (L2-based negatives):")
        print(f"    L2 range: [{min(negative_no_collision_l2):.3f}, {max(negative_no_collision_l2):.3f}] (all > {threshold_display:.3f})")
    if pos_l2:
        threshold_display = collision_threshold_good if mean_l2 is not None and std_l2 is not None else threshold_good
        print(f"\n  Category 1 details (positive):")
        print(f"    L2 range: [{min(pos_l2):.3f}, {max(pos_l2):.3f}] (all < {threshold_display:.3f})")
    
    return pos_feat, neg_feat


def main():
    parser = argparse.ArgumentParser()
    # Defaults target Bench2DriveZoo VAD-base on B2D (adjust as needed)
    parser.add_argument(
        '--config',
        default=str(B2D_ZOO_ROOT / 'adzoo/vad/configs/VAD/VAD_base_e2e_b2d.py'),
        help='VAD config path (Bench2DriveZoo).'
    )
    parser.add_argument(
        '--checkpoint',
        default=str(B2D_ZOO_ROOT / 'ckpts/vad_b2d_base.pth'),
        help='VAD checkpoint path (download into Bench2DriveZoo/ckpts).'
    )
    parser.add_argument(
        '--json',
        default=str(B2D_ZOO_ROOT / 'data/infos/b2d_repair_collect_tiny_traj.json'),
        help='Collected per-frame JSON for repair (generated by adzoo/vad/test.py --collect-data).'
    )
    parser.add_argument('--layers', nargs='+', 
                       default=['pts_bbox_head.ego_fut_decoder.0', 'pts_bbox_head.ego_fut_decoder.2'],
                       help='List of layer names to repair (space-separated)')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='For Arachne_v1: Alpha coefficient for threshold calculation (default: 0.5). For semSegRep: Ignored (always uses median threshold)')
    parser.add_argument('--rep-method', choices=['Arachne_v1', 'Arachne_v2', 'semSegRep'],
                       default='Arachne_v1',
                       help='Repair method: Arachne_v1 (statistical thresholds, GL+FI on negative only), '
                            'Arachne_v2 (changed/unchanged method with cost_chgd/(1+cost_unchgd)), '
                            'or semSegRep (fixed threshold baseline)')
    parser.add_argument('--search-algo', choices=['PSO', 'DE'], default='PSO',
                       help='Search algorithm: PSO (Particle Swarm Optimization) or DE (Differential Evolution)')
    
    parser.add_argument('--num-particles', type=int, default=100)
    parser.add_argument('--num-iterations', type=int, default=20)
    parser.add_argument('--early-stop-patience', type=int, default=5,
                       help='Early stopping patience: stop if fitness does not improve for N iterations (default: 5, enabled). Set to 0 or None to disable. Example: --early-stop-patience 5')
    parser.add_argument('--output-dir', default='./repair_both_layers')
    parser.add_argument('--use-vad-eval', action='store_true',
                       help='Use actual VAD evaluation for fitness (slow but accurate)')
    parser.add_argument('--num-weights-to-repair', type=int, default=None,
                       help='Target number of weights to repair (uses multi-layer Pareto if needed)')
    parser.add_argument('--fitness-type', type=str, default='discrete',
                       choices=['discrete', 'continuous', 'continuous2'],
                       help='Fitness function type: discrete (weighted frame counts), continuous (total L2 error), '
                            'or continuous2 (total L2 + collision penalty)')
    parser.add_argument('--time-horizon', type=int, choices=[1, 2, 3], default=3,
                       help='Time horizon for L2 error and collision: 1s (2 timesteps), 2s (4 timesteps), or 3s (6 timesteps, default). '
                            'This affects both L2 error computation and collision detection (plan_obj_box_col_Xs).')
    args = parser.parse_args()
    
    # Check alpha parameter validity based on repair method
    if args.rep_method in ['Arachne_v1', 'Arachne_v2']:
        if args.alpha <= 0:
            parser.error(f"--alpha must be > 0 for {args.rep_method}, current value: {args.alpha}")
    elif args.rep_method == 'semSegRep':
        # semSegRep always uses median threshold, alpha parameter is ignored
        print(f"  [INFO] semSegRep: --alpha parameter ({args.alpha}) is ignored, will use median threshold")
    
    print("="*70)
    print(f"Repair Method: {args.rep_method}")
    print(f"Repair VAD layers: {', '.join(args.layers)}")
    print("="*70)
    
    # Load model
    print("\n[1/4] Loading VAD model...")
    vad_model, cfg = load_vad_model(args.config, args.checkpoint)
    
    # Extract target layers
    print("\nTarget layers:")
    target_layers = {}
    total_params = 0
    for layer_name in args.layers:
        try:
            layer = vad_model.get_submodule(layer_name)
            if not isinstance(layer, nn.Linear):
                print(f"  Error: Layer '{layer_name}' is not Linear!")
                continue
            target_layers[layer_name] = layer
            layer_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            total_params += layer_params
            print(f"  {layer_name}: {layer.weight.shape} ({layer_params} params)")
        except AttributeError:
            print(f"  Error: Layer '{layer_name}' not found!")
    
    print(f"  Total: {total_params} parameters across {len(args.layers)} layer(s)")
    
    # Load data
    print("\n[2/4] Loading data...")
    with open(args.json, 'r') as f:
        data_infos = json.load(f)
    
    # Calculate thresholds based on repair method
    # Also count total valid frames in JSON (for later validation)
    # Use the corresponding L2 field based on time_horizon
    l2_field = f'plan_L2_{args.time_horizon}s'
    total_valid_frames_in_json = 0
    for info in data_infos:
        if info.get('fut_valid_flag', False) and info.get(l2_field) is not None:
            total_valid_frames_in_json += 1
    
    if args.rep_method in ['Arachne_v1', 'Arachne_v2']:
        # Automatically calculate thresholds: compute L2 error statistics by actually running the model
        # This ensures thresholds are based on the same L2 values used for classification
        print("\nComputing L2 error statistics to automatically determine thresholds...")
        print("  (Computing L2 errors by running model forward pass, not using pre-computed values)")
        vad_model.eval()
        l2_errors = []
        
        with torch.no_grad():
            for info in data_infos:
                if not info.get('fut_valid_flag', False):
                    continue
                
                # Load ego_features
                ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
                
                # Forward through decoder to get prediction
                pred_traj = vad_model.pts_bbox_head.ego_fut_decoder(ego_features.unsqueeze(0))
                
                # Compute L2 error (same method as in extract_features)
                gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)  # [T, 2]
                cmd_idx = info.get('ego_fut_cmd_idx', 0)
                
                ego_fut_mode = int(getattr(vad_model.pts_bbox_head, 'ego_fut_mode', 6))
                fut_ts = int(getattr(vad_model.pts_bbox_head, 'valid_fut_ts', 6))
                pred_traj = pred_traj.view(ego_fut_mode, fut_ts, 2)
                cmd_idx = int(np.clip(cmd_idx, 0, ego_fut_mode - 1))
                pred_mode = pred_traj[cmd_idx]  # [fut_ts, 2] deltas
                
                # IMPORTANT: VAD decoder outputs delta (displacement) values, not absolute positions
                # Convert delta to absolute positions using cumsum before computing L2 error
                pred_mode_absolute = torch.cumsum(pred_mode, dim=0)  # Convert delta to absolute positions
                
                # Compute L2 error for the specified time horizon
                # Each timestep = 0.5s: 1s=2 points, 2s=4 points, 3s=6 points (full trajectory)
                # Dictionary mapping: {time_horizon_in_seconds: number_of_timesteps}
                timesteps_for_horizon = {1: 2, 2: 4, 3: 6}  # {1秒: 2步, 2秒: 4步, 3秒: 6步}
                num_timesteps = timesteps_for_horizon.get(args.time_horizon, 6)
                min_len = min(num_timesteps, int(pred_mode_absolute.shape[0]), int(gt_traj.shape[0]))
                l2_error = torch.norm(pred_mode_absolute[:min_len] - gt_traj[:min_len], dim=1).mean().item()
                
                if not np.isnan(l2_error) and not np.isinf(l2_error):
                    l2_errors.append(float(l2_error))
        
        if len(l2_errors) == 0:
            parser.error("Error: No valid L2 error data found! Cannot automatically calculate thresholds.")
        
        l2_errors = np.array(l2_errors)
        mean_l2 = float(np.mean(l2_errors))
        std_l2 = float(np.std(l2_errors))
        
        threshold_good = mean_l2 - args.alpha * std_l2
        threshold_bad = mean_l2 + args.alpha * std_l2
        
        print(f"  L2 error statistics: mean = {mean_l2:.6f}, std = {std_l2:.6f} (from {len(l2_errors)} frames)")
        print(f"  Using alpha = {args.alpha}")
        print(f"  Automatically calculated thresholds (for VAD evaluation and classification):")
        print(f"    threshold_good = mean - alpha * std = {mean_l2:.6f} - {args.alpha} * {std_l2:.6f} = {threshold_good:.6f}")
        print(f"    threshold_bad = mean + alpha * std = {mean_l2:.6f} + {args.alpha} * {std_l2:.6f} = {threshold_bad:.6f}")
        mean_l2_for_classification = mean_l2
        std_l2_for_classification = std_l2
    elif args.rep_method == 'semSegRep':
        mean_l2_for_classification = None
        std_l2_for_classification = None
        # semSegRep ALWAYS uses median threshold, regardless of --alpha parameter
        print("\nUsing automatic median threshold for semSegRep (--alpha parameter is ignored)...")
        vad_model.eval()
        l2_errors = []
        with torch.no_grad():
            for info in data_infos:
                if not info.get('fut_valid_flag', False):
                    continue
                ego_features = torch.tensor(info['ego_features'], dtype=torch.float32)
                pred_traj = vad_model.pts_bbox_head.ego_fut_decoder(ego_features.unsqueeze(0))
                gt_traj = torch.tensor(info['ground_truth'], dtype=torch.float32)
                cmd_idx = info.get('ego_fut_cmd_idx', 0)
                ego_fut_mode = int(getattr(vad_model.pts_bbox_head, 'ego_fut_mode', 6))
                fut_ts = int(getattr(vad_model.pts_bbox_head, 'valid_fut_ts', 6))
                pred_traj = pred_traj.view(ego_fut_mode, fut_ts, 2)
                cmd_idx = int(np.clip(cmd_idx, 0, ego_fut_mode - 1))
                pred_mode = pred_traj[cmd_idx]
                pred_mode_absolute = torch.cumsum(pred_mode, dim=0)
                # Compute L2 error for the specified time horizon
                # Each timestep = 0.5s: 1s=2 points, 2s=4 points, 3s=6 points (full trajectory)
                # Dictionary mapping: {time_horizon_in_seconds: number_of_timesteps}
                timesteps_for_horizon = {1: 2, 2: 4, 3: 6}  # {1秒: 2步, 2秒: 4步, 3秒: 6步}
                num_timesteps = timesteps_for_horizon.get(args.time_horizon, 6)
                min_len = min(num_timesteps, int(pred_mode_absolute.shape[0]), int(gt_traj.shape[0]))
                l2_error = torch.norm(pred_mode_absolute[:min_len] - gt_traj[:min_len], dim=1).mean().item()
                if not np.isnan(l2_error) and not np.isinf(l2_error):
                    l2_errors.append(float(l2_error))
        if len(l2_errors) == 0:
            parser.error("Error: No valid L2 data found to compute median threshold for semSegRep.")
        median_l2 = float(np.median(np.array(l2_errors)))
        threshold_good = median_l2
        threshold_bad = median_l2
        print(f"  Computed median L2 threshold: {median_l2:.6f}")
        print("  Classification: L2 < median = positive, L2 >= median = negative")
        print(f"  Note: --alpha parameter ({args.alpha}) was ignored for semSegRep")
    
    # Pre-check: warn if thresholds are unreasonable
    # Note: Since alpha > 0, threshold_bad is always greater than threshold_good, no need to check
    if threshold_good < 0:
        print(f"\n  WARNING: threshold_good ({threshold_good:.6f}) < 0, may result in no positive frames!")
    
    # Extract features with classification based on repair method
    if args.rep_method == 'Arachne_v1':
        # Classification uses collision + mean ± alpha*std logic
        pos_feat, neg_feat = extract_features(
            vad_model, data_infos, threshold_good, threshold_bad,
            mean_l2=mean_l2_for_classification,
            std_l2=std_l2_for_classification,
            alpha=args.alpha,
            rep_method='Arachne_v1',
            time_horizon=args.time_horizon
        )
    elif args.rep_method == 'Arachne_v2':
        # Arachne_v2 uses the same classification logic as Arachne_v1
        # The difference is in localization: v2 uses changed/unchanged method
        pos_feat, neg_feat = extract_features(
            vad_model, data_infos, threshold_good, threshold_bad,
            mean_l2=mean_l2_for_classification,
            std_l2=std_l2_for_classification,
            alpha=args.alpha,
            rep_method='Arachne_v1',  # Use same classification logic as v1
            time_horizon=args.time_horizon
        )
    elif args.rep_method == 'semSegRep':
        # For semSegRep: simple fixed threshold classification (no collision, no statistics)
        pos_feat, neg_feat = extract_features(
            vad_model, data_infos, threshold_good, threshold_bad,
            mean_l2=None,
            std_l2=None,
            alpha=args.alpha,
            rep_method='semSegRep',
            time_horizon=args.time_horizon
        )
    
    # Check if there are enough positive/negative frames
    if len(pos_feat) == 0:
        error_msg = f"\nError: No positive frames found!"
        if args.rep_method == 'semSegRep':
            error_msg += "\nSuggestion: Median threshold may be too strict. Check your data distribution."
        else:
            error_msg += f"\nSuggestion: Decrease --alpha value (current: {args.alpha})"
        parser.error(error_msg)
    
    if len(neg_feat) == 0:
        error_msg = f"\nError: No negative frames found!"
        if args.rep_method == 'semSegRep':
            error_msg += "\nSuggestion: Median threshold may be too lenient. Check your data distribution."
        else:
            error_msg += f"\nSuggestion: Increase --alpha value (current: {args.alpha})"
        parser.error(error_msg)
    
    if len(pos_feat) < 10:
        print(f"\nWarning: Only {len(pos_feat)} positive samples (recommended at least 10)!")
        if args.rep_method == 'semSegRep':
            print("Suggestion: Median threshold may be too strict. Check your data distribution.")
        else:
            print(f"Suggestion: Decrease --alpha value (current: {args.alpha})")
    
    if len(neg_feat) < 10:
        print(f"\nWarning: Only {len(neg_feat)} negative samples (recommended at least 10)!")
        if args.rep_method == 'semSegRep':
            print("Suggestion: Median threshold may be too lenient. Check your data distribution.")
        else:
            print(f"Suggestion: Increase --alpha value (current: {args.alpha})")
    
    # Print frame statistics for GL/FI calculation
    print("\n" + "="*70)
    print("Frame Statistics for Localization (GL/FI)")
    print("="*70)
    print(f"  Positive frames (for maintaining correctness): {len(pos_feat)}")
    print(f"  Negative frames (for fixing errors): {len(neg_feat)}")
    print(f"  Total frames: {len(pos_feat) + len(neg_feat)}")
    print("="*70)
    
    # Stack features and create dummy labels (arachne_pytorch expects tuple (features, labels))
    if pos_feat:
        pos_feat_dim = pos_feat[0].shape[0]
        input_pos = (torch.stack(pos_feat), torch.empty(len(pos_feat), 1))
    else:
        # Need to get feature dimension from neg_feat or data_infos
        if neg_feat:
            pos_feat_dim = neg_feat[0].shape[0]
        elif data_infos:
            pos_feat_dim = len(data_infos[0].get('ego_features', []))
        else:
            pos_feat_dim = 512  # Default fallback
        input_pos = (torch.empty(0, pos_feat_dim), torch.empty(0, 1))
    
    if neg_feat:
        neg_feat_dim = neg_feat[0].shape[0]
        input_neg = (torch.stack(neg_feat), torch.empty(len(neg_feat), 1))
    else:
        # Need to get feature dimension from pos_feat or data_infos
        if pos_feat:
            neg_feat_dim = pos_feat[0].shape[0]
        elif data_infos:
            neg_feat_dim = len(data_infos[0].get('ego_features', []))
        else:
            neg_feat_dim = 512  # Default fallback
        input_neg = (torch.empty(0, neg_feat_dim), torch.empty(0, 1))
    
    print(f"\n[3/4] Creating wrapper for {len(args.layers)} layer(s)...")
    
    # Get the full decoder
    full_decoder = vad_model.pts_bbox_head.ego_fut_decoder
    
    # Create wrapper
    wrapper = MultiLayerWrapper(target_layers, full_decoder)
    
    print("Wrapper created:")
    print(f"  Target layers to repair: {list(target_layers.keys())}")
    print(f"  Full decoder length: {len(full_decoder)} layers")
    print(f"  Trainable layer names: {wrapper.layer_names}")
    
    # Run Arachne
    print(f"\n[4/4] Running Arachne on {len(args.layers)} layer(s)...")
    
    arachne = ArachnePyTorch()
    arachne.set_options(
        num_particles=args.num_particles,
        num_iterations=args.num_iterations,
        num_weights_to_repair=args.num_weights_to_repair,
        target_layers=wrapper.layer_names,  # Pass wrapper layer names to arachne
        early_stop_patience=args.early_stop_patience,
        optimization_algorithm=args.search_algo  # Set search algorithm (PSO or DE)
    )
    
    if args.num_weights_to_repair:
        print(f"  Target number of weights to repair: {args.num_weights_to_repair}")
    
    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Localize
    print("\n" + "="*70)
    print("Localizing faulty weights...")
    print("="*70)
    
    # Determine if we should use Arachne v2 (changed/unchanged method)
    use_changed_unchanged = (args.rep_method == 'Arachne_v2')
    
    if use_changed_unchanged:
        print(f"Using Arachne v2 Changed/Unchanged method:")
        print(f"  Changed (negative) frames: {len(neg_feat)}")
        print(f"  Unchanged (positive) frames: {len(pos_feat)}")
        print(f"  Will compute GL and FI separately for changed and unchanged samples,")
        print(f"  then combine using cost_chgd / (1 + cost_unchgd)")
        weights = arachne.localize(
            wrapper, input_neg, input_pos=input_pos,
            output_dir=output_dir, use_changed_unchanged=True
        )
    else:
        print(f"Using {len(neg_feat)} negative frames for Gradient Loss (GL) and Forward Impact (FI) calculation")
        weights = arachne.localize(wrapper, input_neg, output_dir=output_dir)
    
    print(f"\nFound {len(weights)} weights to repair")
    
    # Build frame data for evaluation
    frame_data_dict = None
    positive_frames = None
    negative_frames = None
    if args.use_vad_eval:
        print("Building frame_data_dict with frame_identifiers=None (should get all frames)...")
        frame_data_dict = build_frame_data_dict(args.json, frame_identifiers=None)
        
        # For Arachne_v1 and Arachne_v2: positive_frames and negative_frames are not used in evaluation
        # (evaluation uses ALL frames in frame_data_dict with threshold-based classification)
        # But we still need to pass them for API compatibility, so use None or empty lists
        # Note: The actual classification for evaluation uses the same thresholds but evaluates ALL frames
        if args.rep_method in ['Arachne_v1', 'Arachne_v2']:
            # For Arachne_v1/v2, we don't need separate frame identifiers since evaluation
            # uses ALL frames in frame_data_dict with threshold-based classification
            positive_frames = None
            negative_frames = None
            print(f"\nNote: For {args.rep_method}, evaluation uses ALL frames in frame_data_dict")
            print(f"  with threshold-based classification (same as localization thresholds)")
            print(f"  Localization used: {len(pos_feat)} positive, {len(neg_feat)} negative frames")
            if args.rep_method == 'Arachne_v2':
                print(f"    (Arachne_v2: using changed/unchanged method for localization)")
            else:
                print(f"    (classified using collision + statistical thresholds)")
        else:
            # For semSegRep, use simple threshold-based extraction
            positive_frames, negative_frames = extract_frame_identifiers(
                args.json, 
                threshold_good=threshold_good, 
                threshold_bad=threshold_bad
            )
            print(f"\nData for evaluation:")
            print(f"  Positive frames: {len(positive_frames)} (L2 < {threshold_good:.6f})")
            print(f"  Negative frames: {len(negative_frames)} (L2 > {threshold_bad:.6f})")
        
        print(f"\nData for evaluation (all frames):")
        print(f"  Total frames in dict: {len(frame_data_dict)}")
        
        # Critical check: frame_data_dict must not be empty
        if len(frame_data_dict) == 0:
            print(f"\nERROR: frame_data_dict is EMPTY! Cannot proceed with evaluation.")
            print(f"  JSON file: {args.json}")
            print(f"  JSON file contains {total_valid_frames_in_json} valid frames in total")
            print("\nPossible causes:")
            print("  1. JSON file is empty or corrupted")
            print("  2. All frames missing required fields (ego_features, ground_truth, etc.)")
            print("  3. frame_id matching issue in build_frame_data_dict")
            print("  4. JSON file path is incorrect")
            print("\nSuggestion: Check the JSON file and verify it contains valid frame data.")
            sys.exit(1)
        
        # Sanity check
        if len(frame_data_dict) < 100:
            print(f"\nWARNING: frame_data_dict only has {len(frame_data_dict)} frames!")
            print(f"  JSON file contains {total_valid_frames_in_json} valid frames in total")
            if len(frame_data_dict) < total_valid_frames_in_json * 0.5:
                print(f"  Currently only loaded {len(frame_data_dict)}/{total_valid_frames_in_json} frames ({len(frame_data_dict)/total_valid_frames_in_json*100:.1f}%)")
            print("  This will cause severe overfitting!")
            print("\nPossible causes:")
            print("  1. Missing ground_truth field in JSON")
            print("  2. frame_id matching issue")
            print("  Suggestion: Run python3 diagnose_repair_failure.py for diagnosis")
        
        # Compute original L2 statistics BEFORE optimization
        print("\nComputing original L2 error statistics...")
        # Don't use deepcopy - just evaluate the current wrapper
        # (wrapper will be modified during optimization)
        original_stats = arachne.compute_l2_statistics(
            wrapper, frame_data_dict,
            threshold_good=threshold_good,
            threshold_bad=threshold_bad
        )
    else:
        original_stats = None
    
    # Optimize
    print("\n" + "="*70)
    algo_name = args.search_algo
    print(f"Optimizing weights using {algo_name}...")
    print("="*70)
    if args.fitness_type == 'continuous':
        # For continuous fitness, PSO uses ALL valid frames in frame_data_dict
        total_valid_frames = len(frame_data_dict) if frame_data_dict else 0
        print(f"Fitness type: CONTINUOUS (minimize total L2 error)")
        print(f"Using ALL {total_valid_frames} valid frames for optimization")
        print(f"  (Positive/negative frames are only used for localization, not optimization)")
    elif args.fitness_type == 'continuous2':
        # For continuous2 fitness, PSO uses ALL valid frames in frame_data_dict
        total_valid_frames = len(frame_data_dict) if frame_data_dict else 0
        print(f"Fitness type: CONTINUOUS2 (minimize total L2 error + lambda * N_col)")
        print(f"Using ALL {total_valid_frames} valid frames for optimization")
        print(f"  Formula: total_L2 + lambda * collision_count")
        print(f"  (Positive/negative frames are only used for localization, not optimization)")
    else:
        # For discrete fitness, PSO uses ALL frames but evaluates them differently
        total_valid_frames = len(frame_data_dict) if frame_data_dict else 0
        print(f"Fitness type: DISCRETE (maximize non-negative frames)")
        print(f"Using ALL {total_valid_frames} valid frames for optimization")
        if args.rep_method in ['Arachne_v1', 'Arachne_v2']:
            print(f"  Localization frames (for GL/FI): {len(neg_feat)} negative, {len(pos_feat)} positive")
            if args.rep_method == 'Arachne_v2':
                print(f"    (Arachne_v2: using changed/unchanged method)")
            else:
                print(f"    (Classified using collision check + statistical thresholds: mean ± {args.alpha}*std)")
        else:
            print(f"  Localization frames (for GL/FI): {len(neg_feat)} negative, {len(pos_feat)} positive")
            print(f"    (Classified using fixed threshold: L2 < {threshold_good:.6f} = positive)")
    print("="*70)
    
    repaired, fitness_history = arachne.optimize(
        wrapper, weights, input_neg, input_pos, output_dir, verbose=1,
        use_vad_eval=args.use_vad_eval,
        frame_data_dict=frame_data_dict if args.use_vad_eval else None,
        positive_frames=positive_frames if args.use_vad_eval else None,
        negative_frames=negative_frames if args.use_vad_eval else None,
        threshold_good=threshold_good,
        threshold_bad=threshold_bad,
        fitness_type=args.fitness_type,
        rep_method=args.rep_method,
        time_horizon=getattr(args, 'time_horizon', 3)
    )
    
    # With open-loop evaluation, repaired is always the wrapper model
    repaired_wrapper = repaired
    
    # Compute L2 error statistics before and after optimization
    if args.use_vad_eval and frame_data_dict and original_stats:
        print("\n" + "="*70)
        print("L2 ERROR STATISTICS COMPARISON")
        print("="*70)
        
        print("\nComputing L2 statistics for repaired model...")
        repaired_stats = arachne.compute_l2_statistics(
            repaired_wrapper, frame_data_dict,
            threshold_good=threshold_good,
            threshold_bad=threshold_bad
        )
        
        print("\nOriginal Model:")
        print(f"  Total L2 error: {original_stats['total_l2']:.4f}")
        print(f"  Mean L2 error:  {original_stats['mean_l2']:.4f}")
        print(f"  Median L2 error: {original_stats['median_l2']:.4f}")
        print(f"  Std L2 error:   {original_stats['std_l2']:.4f}")
        print(f"  Min/Max L2:     {original_stats['min_l2']:.4f} / {original_stats['max_l2']:.4f}")
        print(f"  Frame distribution: Pos={original_stats['positive_count']}, Mid={original_stats['middle_count']}, Neg={original_stats['negative_count']}")
        
        print("\nRepaired Model:")
        print(f"  Total L2 error: {repaired_stats['total_l2']:.4f}")
        print(f"  Mean L2 error:  {repaired_stats['mean_l2']:.4f}")
        print(f"  Median L2 error: {repaired_stats['median_l2']:.4f}")
        print(f"  Std L2 error:   {repaired_stats['std_l2']:.4f}")
        print(f"  Min/Max L2:     {repaired_stats['min_l2']:.4f} / {repaired_stats['max_l2']:.4f}")
        print(f"  Frame distribution: Pos={repaired_stats['positive_count']}, Mid={repaired_stats['middle_count']}, Neg={repaired_stats['negative_count']}")
        
        print("\nImprovement:")
        # Calculate change: repaired - original (positive = L2 increased/bad, negative = L2 decreased/good)
        total_change = repaired_stats['total_l2'] - original_stats['total_l2']
        mean_change = repaired_stats['mean_l2'] - original_stats['mean_l2']
        total_change_pct = (total_change / original_stats['total_l2'] * 100) if original_stats['total_l2'] > 0 else 0
        mean_change_pct = (mean_change / original_stats['mean_l2'] * 100) if original_stats['mean_l2'] > 0 else 0
        
        # Display: negative is good (L2 decreased), positive is bad (L2 increased)
        print(f"  Total L2 change: {total_change:+.4f} ({total_change_pct:+.2f}%)")
        print(f"  Mean L2 change:  {mean_change:+.4f} ({mean_change_pct:+.2f}%)")
        print(f"  Positive frames change: {repaired_stats['positive_count'] - original_stats['positive_count']:+d}")
        print(f"  Negative frames change: {repaired_stats['negative_count'] - original_stats['negative_count']:+d}")
        
        if total_change < 0:
            print("\nRepair reduced L2 error!")
        elif total_change > 0:
            print("\nRepair increased L2 error (may need more iterations/particles)")
        else:
            print("\nNo change in L2 error")
    
    # Evaluate repair immediately
    print("\n" + "="*70)
    print("IMMEDIATE EVALUATION (on training data)")
    print("="*70)
    
    # Determine device: check where the model is
    # Check all parameters to ensure model is on a single device
    model_params = list(vad_model.parameters())
    if model_params:
        # Get device from first parameter
        model_device = model_params[0].device
        # Verify all parameters are on the same device
        for param in model_params:
            if param.device != model_device:
                print(f"Warning: Model has parameters on different devices! Found {param.device} vs {model_device}")
                # Move entire model to CPU to avoid device mismatch
                vad_model = vad_model.cpu()
                model_device = torch.device('cpu')
                break
    else:
        model_device = torch.device('cpu')
    
    print(f"Model device: {model_device}")
    
    # Ensure model is on the determined device
    vad_model = vad_model.to(model_device)
    
    # Original model evaluation
    print("\nOriginal model:")
    orig_neg_correct = 0
    orig_pos_correct = 0
    orig_neg_total = len(neg_feat)
    orig_pos_total = len(pos_feat)
    
    with torch.no_grad():
        for feat in neg_feat:
            # Ensure feature is on the same device as model
            # Handle both numpy array and tensor cases
            if isinstance(feat, torch.Tensor):
                feat_on_device = feat.unsqueeze(0).to(model_device)
            else:
                feat_on_device = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(model_device)
            pred = vad_model.pts_bbox_head.ego_fut_decoder(feat_on_device)
            # Simple evaluation: check if prediction is reasonable
            if pred.abs().max() < 10:  # Reasonable prediction
                orig_neg_correct += 1
        
        for feat in pos_feat:
            # Ensure feature is on the same device as model
            # Handle both numpy array and tensor cases
            if isinstance(feat, torch.Tensor):
                feat_on_device = feat.unsqueeze(0).to(model_device)
            else:
                feat_on_device = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(model_device)
            pred = vad_model.pts_bbox_head.ego_fut_decoder(feat_on_device)
            if pred.abs().max() < 10:
                orig_pos_correct += 1
    
    # Repaired model evaluation
    print("\nRepaired model:")
    rep_neg_correct = 0
    rep_pos_correct = 0
    rep_neg_total = len(neg_feat)
    rep_pos_total = len(pos_feat)
    
    # Check repaired wrapper device
    wrapper_params = list(repaired_wrapper.parameters())
    if wrapper_params:
        wrapper_device = wrapper_params[0].device
        # Verify all parameters are on the same device
        for param in wrapper_params:
            if param.device != wrapper_device:
                print(f"Warning: Repaired wrapper has parameters on different devices! Found {param.device} vs {wrapper_device}")
                # Move entire wrapper to model_device to avoid device mismatch
                repaired_wrapper = repaired_wrapper.to(model_device)
                wrapper_device = model_device
                break
    else:
        wrapper_device = model_device
    
    # Ensure wrapper is on the determined device
    repaired_wrapper = repaired_wrapper.to(wrapper_device)
    
    with torch.no_grad():
        for feat in neg_feat:
            # Ensure feature is on the same device as wrapper
            # Handle both numpy array and tensor cases
            if isinstance(feat, torch.Tensor):
                feat_on_device = feat.unsqueeze(0).to(wrapper_device)
            else:
                feat_on_device = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(wrapper_device)
            pred = repaired_wrapper(feat_on_device)
            if pred.abs().max() < 10:
                rep_neg_correct += 1
        
        for feat in pos_feat:
            # Ensure feature is on the same device as wrapper
            # Handle both numpy array and tensor cases
            if isinstance(feat, torch.Tensor):
                feat_on_device = feat.unsqueeze(0).to(wrapper_device)
            else:
                feat_on_device = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(wrapper_device)
            pred = repaired_wrapper(feat_on_device)
            if pred.abs().max() < 10:
                rep_pos_correct += 1
    
    # Print results
    def print_results(name, total, num_correct):
        acc = num_correct / total if total > 0 else 0
        print(f"\n{name}:")
        print(f"  Total: {total} samples")
        print(f"  Correct: {num_correct}/{total} ({acc*100:.1f}%)")
        return acc
    
    orig_neg_acc = print_results("Negative samples (bad predictions)", orig_neg_total, orig_neg_correct)
    orig_pos_acc = print_results("Positive samples (good predictions)", orig_pos_total, orig_pos_correct)
    rep_neg_acc = print_results("Negative samples (bad predictions)", rep_neg_total, rep_neg_correct)
    rep_pos_acc = print_results("Positive samples (good predictions)", rep_pos_total, rep_pos_correct)
    
    # Improvement summary
    print("\n" + "-"*70)
    print("IMPROVEMENT SUMMARY")
    print("-"*70)
    print("Negative samples (should be FIXED):")
    print(f"  Original: {orig_neg_correct}/{orig_neg_total} fixed")
    print(f"  Repaired: {rep_neg_correct}/{rep_neg_total} fixed")
    print(f"  Delta: {rep_neg_correct - orig_neg_correct:+d} samples ({(rep_neg_acc - orig_neg_acc)*100:+.1f}%)")
    
    print("\nPositive samples (should STAY CORRECT):")
    print(f"  Original: {orig_pos_correct}/{orig_pos_total} correct")
    print(f"  Repaired: {rep_pos_correct}/{rep_pos_total} correct")
    print(f"  Delta: {rep_pos_correct - orig_pos_correct:+d} samples ({(rep_pos_acc - orig_pos_acc)*100:+.1f}%)")
    
    total_improvement = (rep_neg_correct - orig_neg_correct) + (rep_pos_correct - orig_pos_correct)
    print(f"\nNet improvement: {total_improvement:+d} samples")
    
    if total_improvement > 0:
        print("Repair improved the model!")
    else:
        print("Repair did not improve (may need more iterations/particles)")
    
    # Copy repaired weights back to original model
    print("\n" + "="*70)
    print("Copying repaired weights to VAD model")
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
    if 'state_dict' in original_checkpoint:
        original_state = original_checkpoint['state_dict']
    else:
        original_state = original_checkpoint
    
    current_state = vad_model.state_dict()
    
    # Build target_layer_keys dynamically from args.layers
    target_layer_keys = []
    for layer_name in args.layers:
        target_layer_keys.append(layer_name + '.weight')
        target_layer_keys.append(layer_name + '.bias')
    
    non_target_keys = [k for k in original_state.keys() if k not in target_layer_keys]
    mismatched = []
    for key in non_target_keys:
        if key not in current_state:
            mismatched.append((key, "missing"))
        elif not torch.equal(original_state[key], current_state[key]):
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
    save_path = output_dir / 'VAD_repaired_both_layers.pth'
    if isinstance(original_checkpoint, dict) and 'state_dict' in original_checkpoint:
        # Original checkpoint has metadata structure
        repaired_checkpoint = original_checkpoint.copy()
        repaired_checkpoint['state_dict'] = vad_model.state_dict()
        torch.save(repaired_checkpoint, save_path)
        print(f"\nSaved repaired model to: {save_path}")
        print(f"  Preserved original checkpoint structure with metadata")
        print(f"  Total keys in state_dict: {len(current_state)}")
        print(f"  Modified keys: {len(target_layer_keys)}")
        print(f"  Unchanged keys: {len(non_target_keys)}")
    else:
        # Original checkpoint is just state_dict
        torch.save(vad_model.state_dict(), save_path)
        print(f"\nSaved repaired model to: {save_path}")
        print(f"  Total keys: {len(current_state)}")
        print(f"  Modified keys: {len(target_layer_keys)}")
        print(f"  Unchanged keys: {len(non_target_keys)}")
    
    # Plot fitness convergence
    if fitness_history and len(fitness_history) > 1:
        print("\n" + "="*70)
        print("FITNESS CONVERGENCE")
        print("="*70)
        print(f"Original fitness (iter 0): {fitness_history[0]:.6f}")
        print(f"Final fitness:             {fitness_history[-1]:.6f}")
        print(f"Improvement:               {fitness_history[0] - fitness_history[-1]:.6f}")
        if fitness_history[0] != 0:
            print(f"Reduction:                 {(1 - fitness_history[-1]/fitness_history[0])*100:.1f}%")
        
        # Plot convergence
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend (no X server needed)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(fitness_history, 'b-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.title('Fitness Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        convergence_path = output_dir / 'fitness_convergence.png'
        plt.savefig(convergence_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nConvergence plot saved to: {convergence_path}")
    
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Repaired {len(weights)} weights across {len(args.layers)} layer(s)")
    print(f"Layers: {', '.join(args.layers)}")
    print(f"Saved to: {save_path}")
    print(f"\nOutput files:")
    print(f"  - {save_path}")
    print(f"  - {output_dir / 'pareto_front.png'}")
    print(f"  - {output_dir / 'fitness_history.json'}")
    print(f"  - {output_dir / 'fitness_convergence.png'}")


if __name__ == '__main__':
    main()