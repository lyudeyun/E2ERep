#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch implementation of Arachne algorithm for VAD repair"""
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from repair_common.arachne_base import ArachneBase
from repair_common.frame_cache import occ_pack_path, TIMESTEPS, VADFrameCache

REPO_ROOT = Path(__file__).resolve().parents[3]

def _resolve_occ_path(occ_path):
    if not occ_path:
        return occ_path
    
    path_str = str(occ_path)

    # If absolute, return as is
    if os.path.isabs(path_str):
        return path_str
    
    # Relative path: resolve against REPO_ROOT
    return str(REPO_ROOT / path_str)


class ArachnePyTorch(ArachneBase):
    """Arachne repair for the VAD ego_fut_decoder."""
    
    def __init__(self):
        super().__init__()
        self._fitness_eval_counter = 0
        self._eval_cache = None  # VADFrameCache of the current frame_data_dict

    def _compute_gradient_loss(self, model, input_neg):
        """
        Compute gradient of the loss w.r.t. weights.
        Supports multi-mode VAD output by selecting the mode specified in target[:, -1].
        """
        model.eval()
        model.zero_grad()
        
        if isinstance(input_neg, (list, tuple)):
            X_neg, y_neg = input_neg
        else:
            # Fallback if just tensor
            X_neg = input_neg
            y_neg = torch.empty(0)

        # Move to device
        if isinstance(X_neg, torch.Tensor):
            X_neg = X_neg.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_neg = torch.tensor(X_neg, dtype=torch.float32).to(self.device)
            
        if isinstance(y_neg, torch.Tensor):
            y_neg = y_neg.clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            y_neg = torch.tensor(y_neg, dtype=torch.float32).to(self.device)

        outputs = model(X_neg)
        
        # --- Gradient Loss Logic ---
        loss = 0.0
        
        # Check if we have valid Ground Truth
        if y_neg.numel() > 0:
            target = y_neg
            
            # Case 1: VAD Multi-mode handling
            # If target has 13 elements (12 coords + 1 cmd_idx) and output has 72 (6*12)
            if target.dim() == 2 and target.shape[1] == 13 and outputs.shape[1] == 72:
                # 1. Split target into coords and cmd_idx
                target_coords = target[:, :12]  # [B, 12]
                cmd_idxs = target[:, 12].long() # [B]
                
                # 2. Reshape output to [B, 6, 12]
                # VAD output is [batch, 6 modes * 6 steps * 2 coords] = [batch, 72]
                batch_size = outputs.shape[0]
                outputs_reshaped = outputs.view(batch_size, 6, 12)
                
                # 3. Select correct mode based on cmd_idx
                # We need to gather along dim 1. 
                # cmd_idxs is [B], make it [B, 1, 12] for gather
                cmd_idxs_expanded = cmd_idxs.view(batch_size, 1, 1).expand(batch_size, 1, 12)
                selected_output = torch.gather(outputs_reshaped, 1, cmd_idxs_expanded).squeeze(1) # [B, 12]
                
                # 4. Compute Loss
                loss_func = torch.nn.L1Loss()
                loss = loss_func(selected_output, target_coords)
                
            # Case 2: Standard matching shapes (or simple reshape)
            else:
                if outputs.shape != target.shape:
                    # Try to view, but warn if sizes don't match
                    if outputs.numel() == target.numel():
                        target = target.view(outputs.shape)
                        loss_func = torch.nn.L1Loss()
                        loss = loss_func(outputs, target)
                    else:
                        # Fallback: Magnitude loss if shapes really don't match
                        # print(f"Warning: GL shape mismatch: Out {outputs.shape} vs Tgt {target.shape}. Using magnitude.")
                        if outputs.dim() == 1:
                            loss = torch.mean(torch.abs(outputs))
                        else:
                            loss = torch.mean(torch.mean(torch.abs(outputs), dim=1))
                else:
                    loss_func = torch.nn.L1Loss()
                    loss = loss_func(outputs, target)
        else:
            # No GT provided: Use output magnitude as proxy loss
            # (Minimizing output magnitude is a heuristic for "suppressing abnormal activations")
            if outputs.dim() == 1:
                loss = torch.mean(torch.abs(outputs))
            else:
                loss = torch.mean(torch.mean(torch.abs(outputs), dim=1))
                
        loss.backward()
        
        return self._collect_target_gradient_candidates(model)

    def _frame_cache(self, frame_data_dict, device):
        key = (id(frame_data_dict), len(frame_data_dict), str(device))
        if self._eval_cache is None or self._eval_cache.key != key:
            self._eval_cache = None
            self._eval_cache = VADFrameCache(frame_data_dict, device, PlanningMetric(), _resolve_occ_path)
        return self._eval_cache

    def _evaluate_fitness_openloop(self, repaired_layers, frame_data_dict,
                                   positive_frames, negative_frames,
                                   threshold_good, threshold_bad, fitness_type='discrete',
                                   rep_method='Arachne_v1', use_original_l2_for_classification=False,
                                   time_horizon=3):
        """
        Fitness of the (repaired) decoder on ALL frames of frame_data_dict (lower is better).

        discrete:    fitness = -(N_pos - N_neg - 0.5 * N_mid - 10 * N_col)
        continuous:  fitness = sum of L2 errors
        continuous2: fitness = sum of L2 errors + 10 * N_col

        N_col counts frames whose plan collides (within the time horizon) at a step where
        the expert does not; the other frames are positive (L2 < threshold_good), middle
        (L2 <= threshold_bad) or negative. For semSegRep with threshold_good ==
        threshold_bad, the threshold is the median of the original L2 in the JSON and
        there are no middle frames.

        All frames are evaluated at once on the GPU (see repair_common/frame_cache.py); the decoder runs
        in chunks of _eval_batch_size frames.

        With use_original_l2_for_classification=True (the one-off evaluation of the
        original model), L2 is taken from the JSON where available and the method
        returns (fitness, frame_counts).
        """
        if fitness_type not in ('discrete', 'continuous', 'continuous2'):
            raise ValueError(f"Unknown fitness_type: {fitness_type}")
        repaired_layers.eval()
        device = next(repaired_layers.parameters()).device if hasattr(repaired_layers, 'parameters') \
            else self.device
        n_steps = TIMESTEPS.get(time_horizon, 6)

        def result(fitness, n_pos, n_mid, n_neg, n_col):
            if use_original_l2_for_classification:
                return float(fitness), {
                    'positive_no_collision': n_pos, 'middle_no_collision': n_mid,
                    'negative_no_collision': n_neg, 'collision': n_col,
                    'total_evaluated': n_pos + n_mid + n_neg + n_col}
            return float(fitness)

        def discrete(n_pos, n_mid, n_neg, n_col):
            return -(1.0 * n_pos - 1.0 * n_neg - 0.5 * n_mid - 10.0 * n_col)

        if len(frame_data_dict) == 0:
            print("  [WARNING] frame_data_dict is empty! No frames to evaluate.")
            return result(float('inf'), 0, 0, 0, 0)
        cache = self._frame_cache(frame_data_dict, device)

        # semSegRep with one threshold: the median of the original L2 in the JSON
        th_good, th_bad = threshold_good, threshold_bad
        single_threshold = rep_method == 'semSegRep' and threshold_good == threshold_bad
        median_found = False
        if single_threshold:
            stored = cache.field(f'plan_L2_{time_horizon}s')
            stored = stored[np.isfinite(stored)]
            if len(stored):
                th_good = th_bad = float(np.median(stored))
                median_found = True
            elif not use_original_l2_for_classification:
                print(f"  [WARNING] semSegRep: Could not compute median L2, using provided threshold {threshold_good}")

        # Original model, methods other than Arachne: classes from the localisation lists
        # and the collision flags stored in the JSON (discrete fitness needs no L2)
        if (use_original_l2_for_classification and fitness_type == 'discrete'
                and rep_method not in ('Arachne_v1', 'Arachne_v2')
                and positive_frames and negative_frames and not (single_threshold and median_found)):
            pos_set, neg_set = set(positive_frames), set(negative_frames)
            keys = list(frame_data_dict.keys())
            col = cache.has_collision
            in_pos = np.array([k in pos_set for k in keys])
            in_neg = np.array([k in neg_set for k in keys])
            n_col = int(col.sum())
            n_pos = int((~col & in_pos).sum())
            n_neg = int((~col & ~in_pos & in_neg).sum())
            n_mid = cache.n - n_col - n_pos - n_neg
            return result(discrete(n_pos, n_mid, n_neg, n_col), n_pos, n_mid, n_neg, n_col)

        batch_size = self._eval_batch_size(device)
        l2_parts, col_parts = [], []
        with torch.no_grad():
            for s in range(0, cache.n, batch_size):
                e = min(s + batch_size, cache.n)
                pred = repaired_layers(cache.feats[s:e])          # (B, 72): 6 commands x 6 steps x 2
                if pred.dim() == 1:
                    pred = pred.unsqueeze(0)
                cmd = cache.cmd[s:e]
                l2_parts.append(self._compute_l2_error_gpu(pred, cache.gt[s:e], cmd, time_horizon=time_horizon))
                pred_abs = self._pred_traj_to_abs_gpu(pred, cmd)[:, :6, :2]
                col_parts.append(cache.collision(pred_abs, s, n_steps))
        l2 = torch.cat(l2_parts).double().cpu().numpy()
        col = torch.cat(col_parts).cpu().numpy()

        if use_original_l2_for_classification:
            stored = cache.field(f'plan_L2_{time_horizon}s')
            l2 = np.where(np.isfinite(stored), stored, l2)

        finite = np.isfinite(l2)
        valid_frame_count = int(finite.sum())
        # Sequential float64 sum in frame order (a Python sum over the list), so that the
        # value does not depend on how the frames are chunked
        total_l2_error = float(sum(l2[finite].tolist()))

        below = l2 < th_good                                       # NaN/inf L2 -> negative
        if single_threshold:
            n_pos = int((~col & below).sum())
            n_mid = 0
            n_neg = int((~col & ~below).sum())
        else:
            within = l2 <= th_bad
            n_pos = int((~col & below).sum())
            n_mid = int((~col & ~below & within).sum())
            n_neg = int((~col & ~below & ~within).sum())
        n_col = int(col.sum())

        if fitness_type == 'discrete':
            fitness = discrete(n_pos, n_mid, n_neg, n_col)
        elif fitness_type == 'continuous':
            fitness = float('inf') if valid_frame_count == 0 else total_l2_error
        else:
            fitness = float('inf') if valid_frame_count == 0 else total_l2_error + 10.0 * n_col

        if use_original_l2_for_classification:
            return result(fitness, n_pos, n_mid, n_neg, n_col)
        self._fitness_eval_counter += 1
        mean_l2 = (total_l2_error / valid_frame_count) if valid_frame_count > 0 else float('inf')
        print(
            f"[fitness-debug] eval#{self._fitness_eval_counter} "
            f"total_l2={total_l2_error:.6f} mean_l2={mean_l2:.6f} "
            f"collision_count={n_col} valid_frames={valid_frame_count}"
        )
        return float(fitness)

    def _compute_l2_error(self, pred_traj, gt_traj, cmd_idx=0):
        """
        Compute L2 error between predicted and ground truth trajectories.
        
        This is used during OPTIMIZATION (PSO/DE) to evaluate fitness.
        
        IMPORTANT: 
        - pred_traj: Decoder output (delta/displacement values, not absolute positions)
        - gt_traj: Ground truth (absolute positions from JSON) - already absolute positions
        
        Parameters
        ----------
        pred_traj : np.ndarray or torch.Tensor
            Predicted trajectory from VAD decoder (DELTA values, not absolute positions)
            Shape: (timesteps, 2) or (batch, timesteps, 2)
        gt_traj : np.ndarray or torch.Tensor
            Ground truth trajectory (ABSOLUTE positions from JSON)
            Shape: (timesteps, 2) or (batch, timesteps, 2)
        cmd_idx : int
            Command index to select (for multi-mode predictions)
        
        Returns
        -------
        l2_error : float
            Average L2 error in meters over all timesteps
        """
        # Convert to numpy if needed
        if isinstance(pred_traj, torch.Tensor):
            pred_traj = pred_traj.detach().cpu().numpy()
        if isinstance(gt_traj, torch.Tensor):
            gt_traj = gt_traj.detach().cpu().numpy()
        
        # Handle multi-mode predictions (select based on cmd_idx)
        if pred_traj.ndim == 3:
            # Shape: (batch, timesteps, 2) or (modes, timesteps, 2)
            if pred_traj.shape[0] > 1:
                # Multiple modes/batches, select by cmd_idx
                pred_traj = pred_traj[cmd_idx] if cmd_idx < pred_traj.shape[0] else pred_traj[0]
            else:
                pred_traj = pred_traj[0]
        
        # IMPORTANT: VAD decoder outputs delta (displacement) values, not absolute positions
        # Convert delta to absolute positions using cumsum (same as VAD.py:434 and test.py:475)
        # This must be done BEFORE comparing with gt_traj (which is already absolute positions)
        if pred_traj.ndim == 2:
            pred_traj = np.cumsum(pred_traj, axis=0)  # Convert delta to absolute positions
        
        # Compute L2 error
        l2_error = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1))
        
        return float(l2_error)

    def _pred_traj_to_abs_gpu(self, pred_traj, cmd_idx):
        """
        Convert raw VAD decoder output to absolute trajectory for collision evaluation.
        Returns shape (batch, 6, 2).
        """
        if pred_traj.dim() == 2:
            if pred_traj.shape[1] == 12:
                pred_traj = pred_traj.view(-1, 6, 2)
            elif pred_traj.shape[1] == 72:
                pred_traj = pred_traj.view(-1, 6, 6, 2)
            else:
                steps = max(1, pred_traj.shape[1] // 2)
                pred_traj = pred_traj.view(-1, steps, 2)

        if pred_traj.dim() == 4:
            if isinstance(cmd_idx, torch.Tensor):
                cmd_idx_clamped = torch.clamp(cmd_idx.long(), 0, pred_traj.shape[1] - 1)
                pred_traj = pred_traj[torch.arange(pred_traj.shape[0], device=pred_traj.device), cmd_idx_clamped]
            else:
                cmd_idx_clamped = max(0, min(pred_traj.shape[1] - 1, int(cmd_idx)))
                pred_traj = pred_traj[:, cmd_idx_clamped]
        elif pred_traj.dim() == 3 and pred_traj.shape[-1] == 2:
            pass
        else:
            raise RuntimeError(f"Unsupported pred_traj shape {tuple(pred_traj.shape)}")

        # VAD decoder outputs delta; convert to absolute
        pred_traj = torch.cumsum(pred_traj, dim=1)
        return pred_traj
    
    def _compute_l2_error_gpu(self, pred_traj, gt_traj, cmd_idx, time_horizon=3):
        """
        Compute L2 error on GPU (faster for batch processing).
        
        Parameters
        ----------
        pred_traj : torch.Tensor
            Predicted trajectory (DELTA values).
            Common shapes:
              - (batch, 72) where 72 = 6 * 6 * 2  (B2D: 6 commands, 6 timesteps)
              - (batch, 6, 6, 2) (explicit)
              - (batch, 6, 2) (already selected command trajectory)
        gt_traj : torch.Tensor
            Ground truth trajectory (ABSOLUTE positions), shape: (batch, 6, 2)
        cmd_idx : int or torch.Tensor
            Command index to select (for multi-mode predictions), shape: (batch,) if Tensor
        time_horizon : int
            Time horizon in seconds: 1s=2 timesteps, 2s=4 timesteps, 3s=6 timesteps (default: 3)
        
        Returns
        -------
        l2_errors : torch.Tensor
            L2 errors for each sample, shape: (batch,)
        """
        batch_size = pred_traj.shape[0]
        # B2D fixed planning horizon: 6 future steps (full trajectory)
        # But we only use the specified time_horizon for L2 calculation
        timesteps_for_horizon = {1: 2, 2: 4, 3: 6}  # {1s: 2 steps, 2s: 4, 3s: 6}
        timesteps = timesteps_for_horizon.get(time_horizon, 6)
        
        # Accept B2D flattened output: (batch, 72) -> (batch, 6, 6, 2) -> select cmd -> (batch, 6, 2)
        # We'll slice to timesteps later after converting to absolute positions
        if pred_traj.dim() == 2:
            if pred_traj.shape[-1] != 72:
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 72) for B2D."
                )
            # Model always outputs full 6 timesteps: reshape to (batch, 6 commands, 6 timesteps, 2)
            pred_traj = pred_traj.view(batch_size, 6, 6, 2)
            # Select command
            if isinstance(cmd_idx, torch.Tensor):
                cmd_idx_clamped = torch.clamp(cmd_idx.long(), 0, 5)
                pred_traj = pred_traj[torch.arange(batch_size, device=pred_traj.device), cmd_idx_clamped]
            else:
                cmd_idx_clamped = max(0, min(5, int(cmd_idx)))
                pred_traj = pred_traj[:, cmd_idx_clamped]
            # Now pred_traj is (batch, 6, 2)
        elif pred_traj.dim() == 4:
            # Explicit: (batch, 6, 6, 2) - model always outputs full 6 timesteps
            if pred_traj.shape[1:] != (6, 6, 2):
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 6, 6, 2) for B2D."
                )
            # Select command
            if isinstance(cmd_idx, torch.Tensor):
                cmd_idx_clamped = torch.clamp(cmd_idx.long(), 0, 5)
                pred_traj = pred_traj[torch.arange(batch_size, device=pred_traj.device), cmd_idx_clamped]
            else:
                cmd_idx_clamped = max(0, min(5, int(cmd_idx)))
                pred_traj = pred_traj[:, cmd_idx_clamped]
            # Now pred_traj is (batch, 6, 2)
        elif pred_traj.dim() == 3:
            # Already selected trajectory: (batch, 6, 2)
            if pred_traj.shape[1:] != (6, 2):
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 6, 2) for selected B2D traj."
                )
        else:
            raise RuntimeError(
                f"Unsupported pred_traj dims {pred_traj.dim()} with shape {tuple(pred_traj.shape)}"
            )
        
        # Ensure pred_traj is (batch, timesteps, 2)
        if pred_traj.dim() == 2:
            # Shape: (batch, 2) - single timestep, add timestep dimension
            pred_traj = pred_traj.unsqueeze(1)
        
        # Convert delta to absolute positions using cumsum
        pred_traj_abs = torch.cumsum(pred_traj, dim=1)
        
        # Ensure gt_traj has correct shape: (batch, 6, 2)
        # GT always has 6 timesteps, we'll slice to timesteps later
        if gt_traj.dim() == 2:
            # Single trajectory: expand to batch
            if gt_traj.shape == (6, 2):
                gt_traj = gt_traj.unsqueeze(0).expand(batch_size, -1, -1)
            else:
                # Reshape and pad if needed
                gt_traj = gt_traj.view(-1, 2).unsqueeze(0)
                if gt_traj.shape[1] != 6:
                    padded_gt = torch.zeros(1, 6, 2, device=gt_traj.device, dtype=gt_traj.dtype)
                    min_timesteps = min(gt_traj.shape[1], 6)
                    padded_gt[:, :min_timesteps, :] = gt_traj[:, :min_timesteps, :]
                    gt_traj = padded_gt.expand(batch_size, -1, -1)
        
        # Slice both pred and gt to timesteps based on time_horizon
        # Both should be (batch, 6, 2) at this point, slice to (batch, timesteps, 2)
        min_len = min(timesteps, pred_traj_abs.shape[1], gt_traj.shape[1])
        if min_len == 0:
            return torch.full((batch_size,), float('inf'), device=pred_traj_abs.device, dtype=pred_traj_abs.dtype)
        pred_traj_abs = pred_traj_abs[:, :min_len, :]
        gt_traj = gt_traj[:, :min_len, :]
        
        # Compute L2 distance for each timestep: (batch, timesteps)
        diff = pred_traj_abs - gt_traj
        l2_distances = torch.sqrt(torch.sum(diff**2, dim=2))  # (batch, timesteps)
        
        # Return average L2 error over specified time horizon: (batch,)
        return torch.mean(l2_distances, dim=1)
    
def build_frame_data_dict(json_file, frame_identifiers=None):
    """
    Build a dictionary of frame data for open-loop evaluation.
    
    Parameters
    ----------
    json_file : str
        Path to VAD evaluation JSON containing ego_features and gt trajectories
    frame_identifiers : list of frame identifiers, optional
        If provided, only include these frames
    
    Returns
    -------
    frame_data_dict : dict
        Dictionary mapping frame_id to frame data
        Each frame contains:
        - 'ego_features': np.ndarray, features before repaired layers
        - 'gt_future_traj': np.ndarray, ground truth trajectory
        - 'plan_L2_3s': float, original L2 error
    """
    # occupancy grids of this JSON, packed into one file (repair_common.frame_cache.pack_occ)
    occ_pack = occ_pack_path(json_file)
    occ_pack = occ_pack if os.path.exists(occ_pack) else None
    print(f"Building frame data dictionary from {json_file}...")
    print(f"  frame_identifiers: {type(frame_identifiers)}, count={len(frame_identifiers) if frame_identifiers else 'None'}")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    frame_data_dict = {}
    frame_id_set = set(frame_identifiers) if frame_identifiers else None
    
    # Debug counters
    skipped_by_filter = 0
    skipped_no_ego = 0
    skipped_no_gt = 0
    
    for idx, frame in enumerate(data):
        # Skip invalid frames (must have fut_valid_flag=True)
        if not frame.get('fut_valid_flag', False):
            continue
        
        # Determine frame identifier (support multiple formats)
        token = frame.get('token')
        scene_name = frame.get('scene_name')
        batch_idx = frame.get('batch_idx')
        frame_idx = frame.get('frame_idx')
        
        if token and scene_name:
            frame_key = (token, scene_name)
        elif batch_idx is not None and frame_idx is not None:
            frame_key = (batch_idx, frame_idx)
        else:
            frame_key = (idx, 0)
        
        # Skip if not in the identifier list
        if frame_id_set and frame_key not in frame_id_set:
            skipped_by_filter += 1
            continue
        
        # Check required fields
        if 'ego_features' not in frame:
            skipped_no_ego += 1
            continue
        
        # Build frame data
        # Save all L2 fields (1s, 2s, 3s) so evaluation can use the appropriate one based on time_horizon
        frame_data = {
            'ego_features': np.array(frame['ego_features'], dtype=np.float32),
            'plan_L2_1s': frame.get('plan_L2_1s', frame.get('plan_L2_3s', 0.0)),  # Fallback to 3s if 1s not available
            'plan_L2_2s': frame.get('plan_L2_2s', frame.get('plan_L2_3s', 0.0)),  # Fallback to 3s if 2s not available
            'plan_L2_3s': frame.get('plan_L2_3s', 0.0),
            'ego_fut_cmd_idx': frame.get('ego_fut_cmd_idx', 0),  # Command index for multi-branch
        }
        
        # Add collision information if available (save all time horizons)
        col_1s = frame.get('plan_obj_box_col_1s', 0.0)
        col_2s = frame.get('plan_obj_box_col_2s', 0.0)
        col_3s = frame.get('plan_obj_box_col_3s', 0.0)
        # Note: has_collision is deprecated, evaluation will use the appropriate collision field based on time_horizon
        frame_data['has_collision'] = (col_1s > 0) or (col_2s > 0) or (col_3s > 0)  # Keep for backward compatibility
        frame_data['plan_obj_box_col_1s'] = float(col_1s) if isinstance(col_1s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_2s'] = float(col_2s) if isinstance(col_2s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_3s'] = float(col_3s) if isinstance(col_3s, (int, float)) else 0.0
        occ_path = frame.get('occ_path', None)
        if occ_path:
            frame_data['occ_path'] = str(occ_path)
            if occ_pack:
                frame_data['occ_pack'] = occ_pack
        
        # Add ground truth trajectory if available
        gt_traj = None
        if 'gt_future_traj' in frame:
            gt_traj = frame['gt_future_traj']
        elif 'gt_ego_fut_trajs' in frame:
            gt_traj = frame['gt_ego_fut_trajs']
        elif 'ground_truth' in frame:
            # Extract from ground_truth field
            gt_data = frame['ground_truth']
            if isinstance(gt_data, dict) and 'fut_traj' in gt_data:
                gt_traj = gt_data['fut_traj']
            elif isinstance(gt_data, list):
                gt_traj = gt_data
        
        if gt_traj is not None:
            frame_data['gt_future_traj'] = np.array(gt_traj, dtype=np.float32)
            frame_data_dict[frame_key] = frame_data
        else:
            skipped_no_gt += 1
    
    print(f"Built frame data dictionary with {len(frame_data_dict)} frames")
    
    # Debug info
    if skipped_by_filter > 0:
        print(f"  Skipped {skipped_by_filter} frames (not in frame_identifiers filter)")
    if skipped_no_ego > 0:
        print(f"  Skipped {skipped_no_ego} frames (no ego_features)")
    if skipped_no_gt > 0:
        print(f"  Skipped {skipped_no_gt} frames (no ground_truth)")
    
    if len(frame_data_dict) == 0:
        print("  Warning: No valid frames found! Check JSON format:")
        print("    Required fields: ego_features, ground_truth (or gt_future_traj)")
        print("    Identifier fields: (token + scene_name) or (batch_idx + frame_idx)")
    elif len(frame_data_dict) < 100:
        print(f"\n  ⚠️  WARNING: Only {len(frame_data_dict)} frames in dict!")
        print(f"  Expected: ~170 frames")
        print(f"  This will cause severe overfitting!")
    
    return frame_data_dict


if __name__ == '__main__':
    print("PyTorch Arachne implementation for VAD repair")
    print("This module provides localize() and optimize() functions")
    print("Example usage:")
    print("""
    from arachne_pytorch import ArachnePyTorch
    from repair_common.arachne_base import load_repair_data
    
    # Load data
    input_neg, input_pos = load_repair_data('vad_complete_data.json')
    
    # Load your PyTorch model
    model = YourVADModel()
    
    # Initialize Arachne
    arachne = ArachnePyTorch()
    arachne.set_options(num_particles=10, num_iterations=20)
    
    # Localize faulty weights
    weights = arachne.localize(model, input_neg, output_dir='./repair_output')
    
    # Optimize and repair
    repaired_model = arachne.optimize(
        model, weights, input_neg, input_pos, output_dir='./repair_output'
    )
    """)
