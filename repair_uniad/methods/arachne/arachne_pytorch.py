#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch implementation of Arachne algorithm for UniAD repair"""
import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import json
from tqdm import tqdm
from mmcv.models.utils.functional import bivariate_gaussian_activation
from repair_common.arachne_base import ArachneBase
from repair_common.frame_cache import TIMESTEPS, UniADFrameCache

REPO_ROOT = Path(__file__).resolve().parents[3]

def _resolve_occ_path(occ_path):
    if not occ_path:
        return occ_path
    
    # Ensure REPO_ROOT is available even in worker processes
    # (Global variables might not be reliable in some multiprocessing contexts)
    repo_root = Path(__file__).resolve().parents[3]

    path_str = str(occ_path)

    if os.path.isabs(path_str):
        return path_str
    
    # Relative path: resolve against repo_root
    return str(repo_root / path_str)


class ArachnePyTorch(ArachneBase):
    """Arachne repair for the UniAD planning_head.reg_branch."""
    
    def __init__(self):
        super().__init__()
        # UniAD planning reg_branch outputs DELTA by default; can be overridden
        self.pred_traj_is_delta = True
        # Apply UniAD planning activation after cumsum (same as planning_head.forward)
        self.apply_bivariate_activation = True
        self._eval_cache = None  # UniADFrameCache of the current frame_data_dict
    
    def set_options(self, **kwargs):
        """Set options for Arachne, including the UniAD trajectory decoding options."""
        super().set_options(**kwargs)
        if 'pred_traj_is_delta' in kwargs:
            self.pred_traj_is_delta = bool(kwargs['pred_traj_is_delta'])
        if 'apply_bivariate_activation' in kwargs:
            self.apply_bivariate_activation = bool(kwargs['apply_bivariate_activation'])
    
    def _compute_gradient_loss(self, model, input_neg):
        """
        Compute Gradient Loss for all weights in target layer(s).
        
        GL(w_ij) = |∂L/∂w_ij|
        
        IMPORTANT: This function uses the RAW decoder output (delta/displacement values),
        NOT absolute positions. This is correct for localization because GL/FI should
        be computed based on the model's raw behavior, not on post-processed values.
        """
        # Convert to tensors (handle both tensor and numpy inputs)
        if isinstance(input_neg[0], torch.Tensor):
            X_neg = input_neg[0].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_neg = torch.tensor(input_neg[0], dtype=torch.float32).to(self.device)
        if isinstance(input_neg[1], torch.Tensor):
            y_neg = input_neg[1].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            y_neg = torch.tensor(input_neg[1], dtype=torch.float32).to(self.device)
        
        # Forward pass
        model.zero_grad()
        outputs = model(X_neg)
        
        # IMPORTANT: outputs here are RAW decoder outputs (delta/displacement values, not absolute positions)
        # This is correct for GL computation - we want to identify weights that contribute
        # to large raw predictions, which correlate with poor trajectory predictions.
        
        # For UniAD trajectory prediction, we use L1Loss (same as UniAD training)
        # The outputs should be [batch, 36] trajectory predictions (delta values)
        
        # For negative samples, we want to minimize the L1 magnitude of predictions
        # This helps identify weights that contribute to large predictions (bad cases)
        # We use L1 norm as proxy for prediction magnitude (consistent with UniAD training)
        if outputs.dim() == 1:
            # If outputs is 1D [batch*36] or just [36], reshape appropriately
            if len(outputs) == 36:
                # Single sample compressed to [36], expand to [1, 36]
                outputs = outputs.unsqueeze(0)
            else:
                # Multiple samples compressed to [batch*36], reshape to [batch, 36]
                batch_size = len(outputs) // 36
                outputs = outputs.reshape(batch_size, 36)
        # MODIFIED: Use L1 Loss against Ground Truth (y_neg) for Gradient Loss
        # This aligns with the official Arachne implementation logic (Gradient of Error).
        
        # Ensure shapes match between outputs and y_neg
        # outputs: [batch, 12] or [batch, 6, 2]
        # y_neg: [batch, 6, 2] (from repair script)
        
        if y_neg.numel() > 0: # Ensure we have valid GT
            target = y_neg
            if outputs.shape != target.shape:
                # Try to reshape target to match outputs
                target = target.view(outputs.shape)
            
            # Compute L1 Loss (mean reduction is standard for gradients)
            loss_func = torch.nn.L1Loss()
            loss = loss_func(outputs, target)
        else:
            # Fallback to magnitude if no GT provided (should not happen with new repair script)
            print("Warning: No Ground Truth labels provided for Gradient Loss. Falling back to Magnitude Loss.")
            if outputs.dim() == 1:
                if len(outputs) == 36:
                    outputs = outputs.unsqueeze(0)
                else:
                    batch_size = len(outputs) // 36
                    outputs = outputs.reshape(batch_size, 36)
            prediction_l1_magnitude = torch.sum(torch.abs(outputs), dim=1)
            loss = torch.mean(prediction_l1_magnitude)
        
        # Backward pass to get gradients
        loss.backward()
        
        return self._collect_target_gradient_candidates(model)
    
    def _frame_cache(self, frame_data_dict, device):
        key = (id(frame_data_dict), len(frame_data_dict), str(device))
        if self._eval_cache is None or self._eval_cache.key != key:
            from mmcv.models.dense_heads.planning_head_plugin import UniADPlanningMetric
            self._eval_cache = None
            self._eval_cache = UniADFrameCache(frame_data_dict, device, UniADPlanningMetric(), _resolve_occ_path)
        return self._eval_cache

    def _evaluate_fitness_openloop(self, repaired_layers, frame_data_dict,
                                   positive_frames, negative_frames,
                                   threshold_good, threshold_bad, fitness_type='discrete',
                                   rep_method='Arachne_v1', use_original_l2_for_classification=False,
                                   time_horizon=3):
        """
        Fitness of the (repaired) reg_branch on ALL frames of frame_data_dict (lower is better).

        discrete:    fitness = -(N_pos - N_neg - 0.5 * N_mid - 10 * N_col)
        continuous:  fitness = sum of L2 errors
        continuous2: fitness = sum of L2 errors + 10 * N_col

        N_col counts frames whose plan collides (within the time horizon) at a step where
        the expert does not; the other frames are positive (L2 < threshold_good), middle
        (L2 <= threshold_bad) or negative. L2 uses the expert's step mask (VAD rule). For
        semSegRep with threshold_good == threshold_bad, the threshold is the median of the
        original L2 in the JSON and there are no middle frames.

        All frames are evaluated at once on the GPU (see repair_common/frame_cache.py); the
        decoder runs in chunks of _eval_batch_size frames.

        With use_original_l2_for_classification=True, L2 and collisions are read from the
        JSON (frames without a stored L2 are skipped) and the model is not run.
        """
        if fitness_type not in ('discrete', 'continuous', 'continuous2'):
            raise ValueError(f"Unknown fitness_type: {fitness_type}")
        repaired_layers.eval()
        device = next(repaired_layers.parameters()).device if hasattr(repaired_layers, 'parameters') \
            else self.device
        n_steps = TIMESTEPS.get(time_horizon, 6)

        def discrete(n_pos, n_mid, n_neg, n_col):
            return -(1.0 * n_pos - 1.0 * n_neg - 0.5 * n_mid - 10.0 * n_col)

        def counts_dict(n_pos, n_mid, n_neg, n_col):
            return {'positive_no_collision': n_pos, 'middle_no_collision': n_mid,
                    'negative_no_collision': n_neg, 'collision': n_col,
                    'total_evaluated': n_pos + n_mid + n_neg + n_col}

        if len(frame_data_dict) == 0:
            print("  [WARNING] frame_data_dict is empty! No frames to evaluate.")
            if use_original_l2_for_classification:
                return float('inf'), counts_dict(0, 0, 0, 0)
            return float('inf')
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
            return discrete(n_pos, n_mid, n_neg, n_col), counts_dict(n_pos, n_mid, n_neg, n_col)

        if use_original_l2_for_classification:
            # Original model: L2 and collisions as stored in the JSON
            l2 = cache.field(f'plan_L2_{time_horizon}s')
            keep = np.isfinite(l2)
            l2 = l2[keep]
            col = (np.nan_to_num(cache.field(f'plan_obj_box_col_{time_horizon}s'), nan=0.0) > 0)[keep]
        else:
            batch_size = self._eval_batch_size(device)
            l2_parts, col_parts = [], []
            with torch.no_grad():
                for s in range(0, cache.n, batch_size):
                    e = min(s + batch_size, cache.n)
                    raw = repaired_layers(cache.feats[s:e])            # (B, 12): 6 steps x 2
                    if raw.dim() == 1:
                        raw = raw.unsqueeze(0)
                    if raw.dim() == 2 and raw.shape[1] != 12:
                        raise RuntimeError(f"Unsupported pred_traj shape {tuple(raw.shape)}. "
                                           f"Expected (batch, 12) for UniAD.")
                    pred = raw.view(-1, 6, 2) if raw.dim() == 2 else raw
                    cmd = cache.cmd[s:e]
                    l2_parts.append(self._compute_l2_error_gpu(pred, cache.gt[s:e], cmd, time_horizon=time_horizon,
                                                               gt_mask=cache.gt_mask[s:e]))
                    pred_abs = self._pred_traj_to_abs_gpu(raw, cmd)[:, :6, :2]
                    col_parts.append(cache.collision(pred_abs, s, n_steps))
            l2 = torch.cat(l2_parts).double().cpu().numpy()
            col = torch.cat(col_parts).cpu().numpy()

        finite = np.isfinite(l2)
        valid_frame_count = int(finite.sum())
        # Sequential float64 sum in frame order (a Python sum over the list), so that the
        # value does not depend on how the frames are chunked
        total_l2_error = float(sum(l2[finite].tolist()))

        below = l2 < th_good                                           # NaN/inf L2 -> negative
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

        if not use_original_l2_for_classification:
            mean_l2 = total_l2_error / valid_frame_count if valid_frame_count > 0 else float('inf')
            print(f"[fitness-debug] fitness={fitness:.6f} total_l2={total_l2_error:.6f} mean_l2={mean_l2:.6f} "
                  f"collision_count={n_col} valid_frames={valid_frame_count}", flush=True)
        return float(fitness)

    def _compute_l2_error(self, pred_traj, gt_traj, cmd_idx=0):
        """
        Compute L2 error between predicted and ground truth trajectories.
        
        This is used during OPTIMIZATION (PSO/DE) to evaluate fitness.
        
        IMPORTANT:
        - pred_traj: Decoder output (delta/displacement values by default for UniAD)
        - gt_traj: Ground truth (absolute positions from JSON) - already absolute positions
        
        Parameters
        ----------
        pred_traj : np.ndarray or torch.Tensor
            Predicted trajectory from UniAD decoder (DELTA by default, can be absolute if pred_traj_is_delta=False)
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
        
        # Convert delta to absolute positions only if needed
        if pred_traj.ndim == 2 and self.pred_traj_is_delta:
            pred_traj = np.cumsum(pred_traj, axis=0)
            if self.apply_bivariate_activation:
                pred_traj = bivariate_gaussian_activation(torch.from_numpy(pred_traj)).numpy()
        
        # Compute L2 error
        l2_error = np.mean(np.linalg.norm(pred_traj - gt_traj, axis=-1))
        
        return float(l2_error)

    def _pred_traj_to_abs_gpu(self, pred_traj, cmd_idx):
        """
        Convert raw UniAD decoder output to absolute trajectory for collision evaluation.
        Returns shape (batch, 6, 2).
        """
        # Handle UniAD flat output: (batch, 12) -> (batch, 6, 2)
        if pred_traj.dim() == 2:
            if pred_traj.shape[1] == 12:
                pred_traj = pred_traj.view(-1, 6, 2)
            else:
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 12) for UniAD."
                )
        elif pred_traj.dim() == 3:
            # Already (batch, 6, 2) or (batch, timesteps, 2)
            if pred_traj.shape[1:] != (6, 2):
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 6, 2) for UniAD."
                )
        else:
            raise RuntimeError(
                f"Unsupported pred_traj dims {pred_traj.dim()} with shape {tuple(pred_traj.shape)}. "
                f"Expected 2D (batch, 12) or 3D (batch, 6, 2) for UniAD."
            )

        if self.pred_traj_is_delta:
            pred_traj = torch.cumsum(pred_traj, dim=1)
            if self.apply_bivariate_activation:
                pred_traj = bivariate_gaussian_activation(pred_traj)

        return pred_traj
    
    def _compute_l2_error_gpu(self, pred_traj, gt_traj, cmd_idx, time_horizon=3, gt_mask=None):
        """
        Compute L2 error on GPU (faster for batch processing).
        
        Parameters
        ----------
        pred_traj : torch.Tensor
            Predicted trajectory (DELTA by default for UniAD).
            Common shapes:
              - (batch, 12) where 12 = 6 * 2  (UniAD: 6 timesteps, 2 coordinates)
              - (batch, 6, 2) (already reshaped)
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
        
        # UniAD layout: (batch, 12) = (batch, 6 timesteps × 2 coords) -> (batch, 6, 2)
        # We'll slice to timesteps later after converting to absolute positions
        if pred_traj.dim() == 2:
            if pred_traj.shape[-1] == 12:
                # UniAD format: (batch, 12) -> (batch, 6, 2)
                pred_traj = pred_traj.view(batch_size, 6, 2)
            else:
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 12) for UniAD."
                )
        elif pred_traj.dim() == 3:
            # Already (batch, 6, 2) format
            if pred_traj.shape[1:] != (6, 2):
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 6, 2) for UniAD."
                )
        else:
            raise RuntimeError(
                f"Unsupported pred_traj dims {pred_traj.dim()} with shape {tuple(pred_traj.shape)}"
            )
        
        # Ensure pred_traj is (batch, timesteps, 2)
        if pred_traj.dim() == 2:
            # Shape: (batch, 2) - single timestep, add timestep dimension
            pred_traj = pred_traj.unsqueeze(1)
        
        # Convert delta to absolute positions only if needed
        pred_traj_abs = torch.cumsum(pred_traj, dim=1) if self.pred_traj_is_delta else pred_traj
        if self.pred_traj_is_delta and self.apply_bivariate_activation:
            pred_traj_abs = bivariate_gaussian_activation(pred_traj_abs)
        
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
        # NOTE: VAD rule requires flipping x coordinate before L2 calculation
        # This matches convert_uniad_to_vad_metrics.py: pred_traj[:, 0] = -pred_traj[:, 0]
        pred_traj_abs_flipped = pred_traj_abs.clone()
        pred_traj_abs_flipped[:, :, 0] = -pred_traj_abs_flipped[:, :, 0]
        gt_traj_flipped = gt_traj.clone()        
        diff = pred_traj_abs_flipped - gt_traj_flipped
        
        # Try without flip
        # diff = pred_traj_abs - gt_traj
        
        # Apply mask if provided (matches VAD rule: l2 = sqrt(((pred - gt) ** 2) * mask).sum(axis=-1))
        if gt_mask is not None:
            # Ensure mask has correct shape: (batch, timesteps, 2)
            # gt_mask should be (batch, 6, 2), slice to (batch, min_len, 2)
            if gt_mask.shape[1] != min_len:
                # Slice mask to match timesteps
                gt_mask = gt_mask[:, :min_len, :]
            # Apply mask: multiply squared diff by mask, then sum over coordinates
            # This matches: l2 = sqrt(((pred - gt) ** 2) * mask).sum(axis=-1) in convert_uniad_to_vad_metrics.py
            # Note: In convert_uniad_to_vad_metrics.py, mask is (6, 2) or (6, 1), and broadcasting happens automatically
            masked_diff_sq = (diff ** 2) * gt_mask  # (batch, timesteps, 2)
            l2_distances = torch.sqrt(torch.sum(masked_diff_sq, dim=2))  # (batch, timesteps)
            
        else:
            # No mask: use standard L2 (assumes all timesteps valid)
            # This should not happen if mask is properly loaded, but fallback for safety
            l2_distances = torch.sqrt(torch.sum(diff**2, dim=2))  # (batch, timesteps)
            
        
        # Return average L2 error over specified time horizon: (batch,)
        # This matches VAD rule: np.mean(l2[:6]) for plan_L2_3s (interval average, not single timestep)
        # JSON format from convert_uniad_to_vad_metrics.py: plan_L2_3s = float(np.mean(l2[:6]))
        # NOTE: JSON uses interval average (mean over timesteps), matching convert_uniad_to_vad_metrics.py:42
        l2_errors = torch.mean(l2_distances, dim=1)
        
        
        
        return l2_errors
    
def build_frame_data_dict(json_file, frame_identifiers=None):
    """
    Build a dictionary of frame data for open-loop evaluation.
    
    Parameters
    ----------
    json_file : str
        Path to UniAD evaluation JSON containing ego_features and gt trajectories
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
        # We only process frames with fut_valid_flag=True throughout the repair process
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
            'fut_valid_flag': frame.get('fut_valid_flag', False),  # Save fut_valid_flag for verification
        }
        
        # Save original predictions from JSON for comparison (if available)
        # This is the prediction used to compute JSON L2, so we can use it for consistency check
        if 'predictions' in frame and isinstance(frame['predictions'], list) and len(frame['predictions']) > 0:
            # predictions[0] is the first mode prediction (absolute format, already cumsum + bivariate activation)
            frame_data['original_predictions'] = np.array(frame['predictions'][0], dtype=np.float32)
        
        # Add collision information if available (save all time horizons)
        col_1s = frame.get('plan_obj_box_col_1s', 0.0)
        col_2s = frame.get('plan_obj_box_col_2s', 0.0)
        col_3s = frame.get('plan_obj_box_col_3s', 0.0)
        # Note: has_collision is deprecated, evaluation will use the appropriate collision field based on time_horizon
        frame_data['has_collision'] = (col_1s > 0) or (col_2s > 0) or (col_3s > 0)  # Keep for backward compatibility
        frame_data['plan_obj_box_col_1s'] = float(col_1s) if isinstance(col_1s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_2s'] = float(col_2s) if isinstance(col_2s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_3s'] = float(col_3s) if isinstance(col_3s, (int, float)) else 0.0
        # Optional: saved occupancy/segmentation for dynamic collision recomputation
        occ_path = frame.get('occ_path', None)
        if occ_path:
            frame_data['occ_path'] = str(occ_path)
        
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
            
            # Add GT mask if available (for VAD rule L2 calculation)
            # VAD rule JSON uses 'ego_fut_masks' (plural), check both for compatibility
            gt_mask = None
            mask_field_used = None
            if 'ego_fut_masks' in frame:  # VAD rule format uses plural
                gt_mask = frame['ego_fut_masks']
                mask_field_used = 'ego_fut_masks'
            elif 'ego_fut_mask' in frame:  # Fallback to singular
                gt_mask = frame['ego_fut_mask']
                mask_field_used = 'ego_fut_mask'
            elif 'gt_mask' in frame:
                gt_mask = frame['gt_mask']
                mask_field_used = 'gt_mask'
            
            if gt_mask is not None:
                # Save as 'ego_fut_mask' (singular) for consistency in frame_data_dict
                # Process mask to ensure correct shape (matches convert_uniad_to_vad_metrics.py)
                gt_mask_array = np.array(gt_mask, dtype=np.float32)
                if gt_mask_array.ndim == 3:
                    gt_mask_array = gt_mask_array[0, :6, :2]
                elif gt_mask_array.ndim == 2:
                    gt_mask_array = gt_mask_array[:6, :2] if gt_mask_array.shape[1] >= 2 else gt_mask_array[:6, None]
                elif gt_mask_array.ndim == 1:
                    gt_mask_array = gt_mask_array[:6, None]
                # Ensure (6, 2) shape
                if gt_mask_array.shape[0] < 6:
                    padded = np.ones((6, 2), dtype=np.float32)
                    if gt_mask_array.ndim == 1:
                        padded[:gt_mask_array.shape[0], :] = gt_mask_array[:gt_mask_array.shape[0], None]
                    else:
                        min_dim = min(gt_mask_array.shape[1] if gt_mask_array.ndim > 1 else 1, 2)
                        padded[:gt_mask_array.shape[0], :min_dim] = gt_mask_array[:gt_mask_array.shape[0], :min_dim]
                    gt_mask_array = padded
                elif gt_mask_array.shape[0] >= 6:
                    if gt_mask_array.ndim == 1:
                        gt_mask_array = np.column_stack([gt_mask_array[:6], gt_mask_array[:6]])
                    elif gt_mask_array.ndim == 2:
                        if gt_mask_array.shape[1] < 2:
                            gt_mask_array = np.column_stack([gt_mask_array[:6, 0], gt_mask_array[:6, 0]])
                        else:
                            gt_mask_array = gt_mask_array[:6, :2]
                frame_data['ego_fut_mask'] = gt_mask_array
                # Also save original field name for debugging
                frame_data['ego_fut_masks'] = gt_mask_array
                frame_data['_mask_field_used'] = mask_field_used  # Debug: which field was used
            else:
                # No mask found - this is expected for some JSON formats
                frame_data['_mask_field_used'] = None  # Debug: no mask found
            
            frame_data_dict[frame_key] = frame_data
        else:
            skipped_no_gt += 1
    
    print(f"Built frame data dictionary with {len(frame_data_dict)} frames")
    
    # Debug: Count frames with/without mask
    frames_with_mask = sum(1 for fd in frame_data_dict.values() if fd.get('ego_fut_mask') is not None)
    frames_without_mask = len(frame_data_dict) - frames_with_mask
    print(f"  Frames with mask: {frames_with_mask}/{len(frame_data_dict)}")
    print(f"  Frames without mask: {frames_without_mask}/{len(frame_data_dict)} (will use default mask=1)")
    
    # Debug: Verify all frames have fut_valid_flag=True
    frames_with_valid_flag = sum(1 for fd in frame_data_dict.values() if fd.get('fut_valid_flag', False))
    print(f"  Frames with fut_valid_flag=True: {frames_with_valid_flag}/{len(frame_data_dict)}")
    if frames_with_valid_flag != len(frame_data_dict):
        print(f"  [WARNING] Some frames in frame_data_dict do not have fut_valid_flag=True!")
    
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
    print("PyTorch Arachne implementation for UniAD repair")
    print("This module provides localize() and optimize() functions")
    print("Example usage:")
    print("""
    from arachne_pytorch import ArachnePyTorch
    from repair_common.arachne_base import load_repair_data
    
    # Load data
    input_neg, input_pos = load_repair_data('vad_complete_data.json')
    
    # Load your PyTorch model
    model = YourUniADModel()
    
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
