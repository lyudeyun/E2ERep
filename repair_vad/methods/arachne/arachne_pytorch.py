#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch implementation of Arachne algorithm for VAD repair"""
import torch
import torch.nn as nn
import numpy as np
import os
import multiprocessing as mp
from pathlib import Path
import json
from tqdm import tqdm
from mmcv.models.dense_heads.planning_head_plugin.metric_stp3 import PlanningMetric
from repair_common.arachne_base import ArachneBase

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------
# Collision helpers for multiprocessing
# ---------------------------------------------------------------------
_COLLISION_METRIC = None


def _init_collision_worker():
    global _COLLISION_METRIC
    if _COLLISION_METRIC is None:
        _COLLISION_METRIC = PlanningMetric()


def _compute_collision_from_occ(args):
    """
    Args:
        args: (occ_path, pred_abs_np, gt_abs_np, time_horizon)
    Returns:
        (ok: bool, result_or_error: bool|str)
    """
    try:
        occ_path, pred_abs_np, gt_abs_np, time_horizon = args
        occ_path = _resolve_occ_path(occ_path)
        if not occ_path or not os.path.exists(occ_path):
            return False, f"Missing occ_path for collision recomputation: {occ_path}"
        metric = _COLLISION_METRIC or PlanningMetric()
        occ_np = np.load(occ_path)['occ']
        occ_t = torch.from_numpy(occ_np)
        if occ_t.dim() == 3:
            occ_t = occ_t.unsqueeze(0)
        pred_t = torch.from_numpy(pred_abs_np).unsqueeze(0)
        gt_t = torch.from_numpy(gt_abs_np).unsqueeze(0)
        _, obj_box_coll = metric.evaluate_coll(pred_t, gt_t, occ_t)
        if time_horizon == 1:
            col_value = float(obj_box_coll[:2].mean().item())
        elif time_horizon == 2:
            col_value = float(obj_box_coll[:4].mean().item())
        else:
            col_value = float(obj_box_coll[:6].mean().item())
        return True, (col_value > 0)
    except Exception as e:
        return False, f"Failed to compute collision from occ_path={occ_path}: {e}"


def _resolve_occ_path(occ_path):
    if not occ_path:
        return occ_path
    
    path_str = str(occ_path)
    
    # Fix legacy path structure: insert /VAD/ if missing
    if "baseline/vad_occ_cache" in path_str and "/VAD/" not in path_str:
        path_str = path_str.replace("baseline/vad_occ_cache", "baseline/VAD/vad_occ_cache")
    
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
    
    def _get_collision_pool(self):
        """Lazily create a process pool for collision evaluation."""
        if self.collision_num_workers is None or self.collision_num_workers <= 1:
            return None
        if self._collision_pool is not None and self._collision_pool_workers == self.collision_num_workers:
            return self._collision_pool
        # Recreate pool if worker count changed
        if self._collision_pool is not None:
            try:
                self._collision_pool.close()
                self._collision_pool.join()
            except Exception:
                pass
        ctx = mp.get_context("spawn")
        self._collision_pool = ctx.Pool(
            processes=self.collision_num_workers,
            initializer=_init_collision_worker
        )
        self._collision_pool_workers = self.collision_num_workers
        return self._collision_pool
    
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

    def _evaluate_fitness_openloop(self, repaired_layers, frame_data_dict,
                                        positive_frames, negative_frames,
                                        threshold_good, threshold_bad, fitness_type='discrete',
                                        rep_method='Arachne_v1', use_original_l2_for_classification=False, time_horizon=3):
        """
        Evaluate fitness using open-loop evaluation (faster).
        
        For discrete fitness:
            fitness = -(w1 * N_pos - w2 * N_(neg-no col) - w3 * N_mid - lambda * N_col)
            where:
                N_pos: number of positive frames (L2 < threshold_good)
                N_(neg-no col): number of negative frames without collision (L2 > threshold_bad, no collision)
                N_mid: number of middle frames (threshold_good <= L2 <= threshold_bad)
                N_col: number of frames with collision
            weights: w1=1.0, w2=1.0, w3=0.5, lambda=10.0
        
        For continuous fitness:
            fitness = total L2 error across all frames
        
        For continuous2 fitness:
            fitness = total L2 error + lambda * N_col
            where:
                total L2 error: sum of L2 errors across all valid frames
                N_col: number of frames with collision
                lambda: penalty coefficient (default 10.0)
        
        Goal: Minimize fitness (maximize the score for discrete, minimize L2 for continuous)
        
        Note: Evaluates ALL frames in frame_data_dict, not just positive_frames and negative_frames
        
        Parameters
        ----------
        repaired_layers : nn.Module or dict
            Repaired layer(s) (wrapper or dict of layers)
        frame_data_dict : dict
            Dictionary containing ALL frames to evaluate
            Each frame should have: 'ego_features', 'gt_trajectory', 'plan_L2_3s'
        positive_frames : list
            List of frames used for localization (not used in evaluation)
        negative_frames : list
            List of frames used for localization (not used in evaluation)
        threshold_good : float
            L2 threshold for positive frames
        threshold_bad : float
            L2 threshold for negative frames
        fitness_type : str
            Type of fitness function ('discrete', 'continuous', 'continuous2')
        rep_method : str
            Repair method name
        
        Returns
        -------
        fitness : float
            Fitness score (lower is better, negative values are good)
        frame_counts : dict, optional
            Dictionary containing frame counts for each category:
            - 'positive_no_collision': number of positive frames without collision
            - 'middle_no_collision': number of middle frames without collision
            - 'negative_no_collision': number of negative frames without collision
            - 'collision': number of frames with collision
            - 'total_evaluated': total number of frames evaluated
            Only returned when use_original_l2_for_classification=True (for logging)
        """
        repaired_layers.eval()
        
        device = repaired_layers.parameters().__next__().device if hasattr(repaired_layers, 'parameters') else self.device
        
        # For semSegRep: if threshold_good == threshold_bad, use median L2 from JSON as threshold
        actual_threshold_good = threshold_good
        actual_threshold_bad = threshold_bad
        use_median_threshold = False
        
        if rep_method == 'semSegRep' and threshold_good == threshold_bad:
            # Calculate median L2 error from original JSON data (plan_L2_Xs based on time_horizon)
            l2_field = f'plan_L2_{time_horizon}s'
            all_original_l2 = []
            for frame_data in frame_data_dict.values():
                original_l2 = frame_data.get(l2_field, None)
                if original_l2 is not None and not np.isnan(original_l2) and not np.isinf(original_l2):
                    all_original_l2.append(original_l2)
            if len(all_original_l2) > 0:
                median_l2 = float(np.median(all_original_l2))
                actual_threshold_good = median_l2
                actual_threshold_bad = median_l2
                use_median_threshold = True
            else:
                if not use_original_l2_for_classification:
                    print(f"  [WARNING] semSegRep: Could not compute median L2, using provided threshold {threshold_good}")
        
        # Count frames in each category
        collision_count = 0
        positive_no_collision_count = 0
        middle_no_collision_count = 0
        negative_no_collision_count = 0
        
        # Debug counters
        error_count = 0
        inf_count = 0
        shape_errors = []
        
        # SIMPLE FIX: For original model evaluation, directly use positive_frames and negative_frames
        # BUT: 
        #   - For Arachne_v1 and Arachne_v2: positive_frames/negative_frames are not used (evaluates ALL frames), so disable fast path
        #   - For semSegRep with median threshold: cannot use fast path because positive_frames/negative_frames
        #     were extracted using a different threshold (0.5) than what we use for evaluation (median L2)
        can_use_fast_path = (use_original_l2_for_classification and 
                            rep_method not in ['Arachne_v1', 'Arachne_v2'] and  # Arachne_v1/v2 don't use positive_frames/negative_frames
                            positive_frames is not None and negative_frames is not None and 
                            len(positive_frames) > 0 and len(negative_frames) > 0 and
                            not (rep_method == 'semSegRep' and use_median_threshold))
        
        if use_original_l2_for_classification:
            if rep_method in ['Arachne_v1', 'Arachne_v2']:
                # Arachne_v1/v2 don't use positive_frames/negative_frames, always use normal evaluation
                pass  # Silent, no warning needed
            elif rep_method == 'semSegRep' and use_median_threshold:
                print(f"  [INFO] semSegRep with median threshold: Cannot use fast path (threshold mismatch), will recompute L2")
            elif positive_frames is None or negative_frames is None:
                print(f"  [WARNING] use_original_l2_for_classification=True but positive_frames={positive_frames is not None}, negative_frames={negative_frames is not None}, falling back to normal evaluation")
            elif len(positive_frames) == 0 or len(negative_frames) == 0:
                print(f"  [WARNING] use_original_l2_for_classification=True but positive_frames={len(positive_frames)}, negative_frames={len(negative_frames)}, falling back to normal evaluation")
            else:
                print(f"  [INFO] Using fast path: directly counting from positive_frames ({len(positive_frames)}) and negative_frames ({len(negative_frames)})")
        
        if can_use_fast_path:
            # Convert to sets for fast lookup
            positive_set = set(positive_frames)
            negative_set = set(negative_frames)
            
            # Count frames directly from frame_data_dict
            for (token, scene_name), frame_data in frame_data_dict.items():
                frame_id = (token, scene_name)
                has_collision = frame_data.get('has_collision', False)
                
                if has_collision:
                    collision_count += 1
                    continue
                
                if frame_id in positive_set:
                    positive_no_collision_count += 1
                elif frame_id in negative_set:
                    negative_no_collision_count += 1
                else:
                    # Middle frames (between thresholds)
                    middle_no_collision_count += 1
            
            # Calculate fitness based on counts
            if fitness_type == 'discrete':
                # Discrete fitness: use same formula as normal evaluation
                # But keep the original sign convention that was working
                w1, w2, w3, lambda_col = 1.0, 1.0, 0.5, 10.0
                score = (w1 * positive_no_collision_count -
                         w2 * negative_no_collision_count -
                         w3 * middle_no_collision_count -
                         lambda_col * collision_count)
                # Keep original sign: fitness = -score (so negative score gives positive fitness)
                fitness = -score
                
                # For semSegRep: calculate median if needed for display
                display_threshold_good = threshold_good
                display_threshold_bad = threshold_bad
                if rep_method == 'semSegRep' and threshold_good == threshold_bad:
                    # Calculate median for display
                    all_original_l2 = []
                    for frame_data in frame_data_dict.values():
                        original_l2 = frame_data.get('plan_L2_3s', None)
                        if original_l2 is not None and not np.isnan(original_l2) and not np.isinf(original_l2):
                            all_original_l2.append(original_l2)
                    if len(all_original_l2) > 0:
                        median_l2 = float(np.median(all_original_l2))
                        display_threshold_good = median_l2
                        display_threshold_bad = median_l2
                
                # Return fitness and frame counts for logging
                if use_original_l2_for_classification:
                    frame_counts = {
                        'positive_no_collision': positive_no_collision_count,
                        'middle_no_collision': middle_no_collision_count,
                        'negative_no_collision': negative_no_collision_count,
                        'collision': collision_count,
                        'total_evaluated': positive_no_collision_count + middle_no_collision_count + negative_no_collision_count + collision_count
                    }
                    return fitness, frame_counts
                return fitness
            else:
                # For continuous fitness, still need to compute L2 errors
                # Fall through to normal evaluation
                pass
        
        # Determine batch size: use configured value or auto-detect based on GPU memory
        # Move outside with torch.no_grad() to ensure it's in function scope
        if self.eval_batch_size is not None:
            batch_size = self.eval_batch_size
        else:
            # Auto-detect batch size based on GPU memory
            if device.type == 'cuda':
                # Get GPU memory info
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                # Conservative estimate: ~50MB per batch (for features + predictions + intermediates)
                # For 80GB GPU (H100): ~1600 theoretical max, use 1024 for better utilization
                # For 32GB GPU: ~640, but use 256 as safe default
                # For 16GB GPU: ~320, but use 128 as safe default
                # For 8GB GPU: ~160, but use 64 as safe default
                if gpu_memory_gb >= 60:
                    batch_size = 1024  # Very large GPU (H100 80GB, etc.)
                elif gpu_memory_gb >= 24:
                    batch_size = 256  # Large GPU (V100, A100, etc.)
                elif gpu_memory_gb >= 12:
                    batch_size = 128  # Medium GPU (RTX 3090, etc.)
                else:
                    batch_size = 64   # Small GPU or default (RTX 3080, T4, etc.)
            else:
                batch_size = 32  # CPU: smaller batch size
        
        frame_items = list(frame_data_dict.items())
        total_frames = len(frame_items)
        
        # Check if frame_data_dict is empty
        if total_frames == 0:
            print(f"  [WARNING] frame_data_dict is empty! No frames to evaluate.")
            if use_original_l2_for_classification:
                frame_counts = {
                    'positive_no_collision': 0,
                    'middle_no_collision': 0,
                    'negative_no_collision': 0,
                    'collision': 0,
                    'total_evaluated': 0
                }
                return float('inf'), frame_counts  # Return worst fitness
            return float('inf')  # Return worst fitness
        
        # Pre-allocate lists for batch processing
        all_ego_features = []
        all_gt_trajectories = []
        all_cmd_indices = []
        all_has_collision = []
        all_frame_data = []  # Store frame_data for later lookup
        
        # Collect all data (same as old version)
        for (token, scene_name), frame_data in frame_items:
            # Convert to numpy arrays (same as old version)
            ego_features = np.array(frame_data['ego_features'], dtype=np.float32)
            gt_trajectory = np.array(frame_data.get('gt_future_traj', []), dtype=np.float32)
            # Use ego_fut_cmd_idx to match old version
            cmd_idx = frame_data.get('ego_fut_cmd_idx', 0)
            all_ego_features.append(ego_features)
            all_gt_trajectories.append(gt_trajectory)
            all_cmd_indices.append(cmd_idx)
            all_has_collision.append(frame_data.get('has_collision', False))
            all_frame_data.append(frame_data)  # Store frame_data for later lookup
        
        # Batch process
        total_l2_error = 0.0
        valid_frame_count = 0
        planning_metric = None
        # Initialize frame count variables
        positive_no_collision_count = 0
        middle_no_collision_count = 0
        negative_no_collision_count = 0
        collision_count = 0
        error_count = 0
        inf_count = 0
        shape_errors = []
        
        # Process batches in no_grad context
        with torch.no_grad():
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_ego_features = all_ego_features[batch_start:batch_end]
                batch_gt_trajectories = all_gt_trajectories[batch_start:batch_end]
                batch_cmd_indices = all_cmd_indices[batch_start:batch_end]
                batch_has_collision = all_has_collision[batch_start:batch_end]
                
                # Convert to tensors in batch (same as old version)
                batch_ego_tensor = torch.tensor(np.array(batch_ego_features), dtype=torch.float32).to(device)
                # Ensure correct shape: (batch, feature_dim)
                if batch_ego_tensor.dim() == 1:
                    batch_ego_tensor = batch_ego_tensor.unsqueeze(0)
                
                # Forward pass
                try:
                    # Batch forward pass (same as old version)
                    batch_pred_traj = repaired_layers(batch_ego_tensor)  # (batch, 36) or (batch, 6, 2)
                    
                    # Ensure pred_trajectories is 2D: (batch, 36) (same as old version)
                    if batch_pred_traj.dim() == 1:
                        batch_pred_traj = batch_pred_traj.unsqueeze(0)
                    elif batch_pred_traj.dim() == 3:
                        # If shape is (batch, 6, 2), flatten to (batch, 12) then pad to (batch, 36) if needed
                        batch_sz = batch_pred_traj.shape[0]
                        if batch_pred_traj.shape[1:] == (6, 2):
                            batch_pred_traj = batch_pred_traj.reshape(batch_sz, 12)
                            # Pad to 36 if needed (shouldn't happen, but handle it)
                            if batch_pred_traj.shape[1] < 36:
                                padding = torch.zeros(batch_sz, 36 - batch_pred_traj.shape[1], device=batch_pred_traj.device)
                                batch_pred_traj = torch.cat([batch_pred_traj, padding], dim=1)
                    
                    # Prepare GT trajectories tensor: (batch, 6, 2)
                    # Always use full 6 timesteps, _compute_l2_error_gpu will slice based on time_horizon
                    batch_gt_tensor = torch.zeros(len(batch_gt_trajectories), 6, 2, dtype=torch.float32, device=device)
                    for i, gt_traj in enumerate(batch_gt_trajectories):
                        if len(gt_traj.shape) == 1:
                            gt_traj = gt_traj.reshape(-1, 2)
                        min_len = min(gt_traj.shape[0], 6)
                        batch_gt_tensor[i, :min_len, :] = torch.tensor(gt_traj[:min_len], dtype=torch.float32, device=device)
                    
                    # Prepare cmd_indices tensor: (batch,)
                    batch_cmd_tensor = torch.tensor(batch_cmd_indices, dtype=torch.long, device=device)
                    
                    # Use _compute_l2_error_gpu which handles all shape cases correctly (same as old version)
                    batch_l2_errors = self._compute_l2_error_gpu(batch_pred_traj, batch_gt_tensor, batch_cmd_tensor, time_horizon=time_horizon)
                    batch_l2_errors_np = batch_l2_errors.cpu().numpy()
                    batch_pred_abs = self._pred_traj_to_abs_gpu(batch_pred_traj, batch_cmd_tensor)

                    # Optional: parallel collision computation on CPU
                    collision_flags = None
                    if self.collision_num_workers is not None and self.collision_num_workers > 1:
                        pool = self._get_collision_pool()
                        tasks = []
                        for b_idx in range(batch_end - batch_start):
                            frame_idx = batch_start + b_idx
                            frame_data = all_frame_data[frame_idx]
                            occ_path = frame_data.get('occ_path', None)
                            pred_abs_np = batch_pred_abs[b_idx:b_idx + 1, :6, :2].detach().cpu().numpy()[0]
                            gt_abs_np = batch_gt_tensor[b_idx:b_idx + 1, :6, :2].detach().cpu().numpy()[0]
                            tasks.append((occ_path, pred_abs_np, gt_abs_np, time_horizon))
                        results = pool.map(_compute_collision_from_occ, tasks)
                        bad = [r for r in results if not r[0]]
                        if bad:
                            error_count += 1
                            shape_errors.append(bad[0][1])
                            continue
                        collision_flags = [r[1] for r in results]
                    
                    # Process each frame in batch for classification and statistics (same as old version)
                    for b_idx in range(batch_end - batch_start):
                        frame_idx = batch_start + b_idx
                        frame_data = all_frame_data[frame_idx]
                        
                        # For original model evaluation, use pre-computed L2 from JSON for consistency
                        # Use the L2 field that matches the time_horizon (plan_L2_1s, plan_L2_2s, or plan_L2_3s)
                        if use_original_l2_for_classification:
                            l2_field = f'plan_L2_{time_horizon}s'
                            original_l2 = frame_data.get(l2_field, None)
                            
                            # Use pre-computed L2 from JSON if available and valid
                            if original_l2 is not None and not np.isnan(original_l2) and not np.isinf(original_l2):
                                l2_error = float(original_l2)
                            else:
                                # Fallback to computed L2 if plan_L2_Xs is missing or invalid
                                l2_error = float(batch_l2_errors_np[b_idx])
                        else:
                            # For particle evaluation, use computed L2 error so fitness reflects model changes
                            l2_error = float(batch_l2_errors_np[b_idx])
                        
                        # Check collision using saved occ; must exist for dynamic collision
                        if collision_flags is not None:
                            has_collision = collision_flags[b_idx]
                        else:
                            occ_path = _resolve_occ_path(frame_data.get('occ_path', None))
                            if not occ_path or not os.path.exists(occ_path):
                                raise ValueError(
                                    f"Missing occ_path for collision recomputation: {occ_path}. "
                                    "Please regenerate JSON with --occ-output-dir."
                                )
                            try:
                                if planning_metric is None:
                                    planning_metric = PlanningMetric()
                                occ_np = np.load(occ_path)['occ']
                                occ_t = torch.from_numpy(occ_np)
                                if occ_t.dim() == 3:
                                    occ_t = occ_t.unsqueeze(0)
                                pred_abs = batch_pred_abs[b_idx:b_idx + 1, :6, :2]
                                gt_abs = batch_gt_tensor[b_idx:b_idx + 1, :6, :2]
                                # Ensure collision inputs are on the same device (CPU is safest here).
                                if pred_abs.device.type != 'cpu' or gt_abs.device.type != 'cpu' or occ_t.device.type != 'cpu':
                                    pred_abs = pred_abs.cpu()
                                    gt_abs = gt_abs.cpu()
                                    occ_t = occ_t.cpu()
                                _, obj_box_coll = planning_metric.evaluate_coll(pred_abs, gt_abs, occ_t)
                                if time_horizon == 1:
                                    col_value = float(obj_box_coll[:2].mean().item())
                                elif time_horizon == 2:
                                    col_value = float(obj_box_coll[:4].mean().item())
                                else:
                                    col_value = float(obj_box_coll[:6].mean().item())
                                has_collision = (col_value > 0)
                            except Exception as e:
                                raise RuntimeError(f"Failed to compute collision from occ_path={occ_path}: {e}")
                        
                        # Track inf values
                        if np.isinf(l2_error) or l2_error == float('inf'):
                            inf_count += 1
                        
                        # Accumulate L2 error for continuous fitness
                        if not np.isnan(l2_error) and not np.isinf(l2_error):
                            valid_frame_count += 1
                            total_l2_error += l2_error
                        
                        # For semSegRep: classify first, then handle collisions
                        # For Arachne_v1: check collision first, skip classification if collision
                        if rep_method == 'semSegRep' and actual_threshold_good == actual_threshold_bad:
                            # Fixed/median threshold: L2 < threshold = positive, L2 >= threshold = negative
                            # Classify first regardless of collision
                            if l2_error < actual_threshold_good:
                                positive_no_collision_count += 1
                            else:
                                negative_no_collision_count += 1
                            
                            # Then handle collision: subtract from corresponding count and add to collision_count
                            if has_collision:
                                if l2_error < actual_threshold_good:
                                    positive_no_collision_count -= 1
                                else:
                                    negative_no_collision_count -= 1
                                collision_count += 1
                        else:
                            # For Arachne_v1: check collision first, skip classification if collision
                            if has_collision:
                                collision_count += 1
                                continue
                            
                            # Classify frame based on L2 error (same logic as localization for non-collision frames)
                            # For original model: uses plan_L2_{time_horizon}s from JSON
                            # For particles: uses computed L2 error (reflects model changes)
                            if l2_error < actual_threshold_good:
                                positive_no_collision_count += 1
                            elif l2_error <= actual_threshold_bad:
                                middle_no_collision_count += 1
                            else:
                                negative_no_collision_count += 1
                
                except Exception as e:
                    error_count += 1
                    shape_errors.append(str(e))
                    continue
            
        # Compute fitness based on type (after all batches are processed)
            if error_count > 0:
                print(f"  [WARNING] {error_count} batch(es) failed during evaluation.")
                if shape_errors:
                    print(f"  [WARNING] Sample error: {shape_errors[0]}")
            total_frames_evaluated = positive_no_collision_count + middle_no_collision_count + negative_no_collision_count + collision_count
            
            if fitness_type == 'discrete':
                # Discrete fitness definition:
                # fitness = -(w1 * N_pos - w2 * N_(neg-no col) - w3 * N_mid - lambda * N_col)
                # We want to maximize this value, but PSO minimizes, so we negate it
                w1 = 1.0
                w2 = 1.0
                w3 = 0.5
                lambda_col = 10.0
                
                score = (w1 * positive_no_collision_count -
                         w2 * negative_no_collision_count -
                         w3 * middle_no_collision_count -
                         lambda_col * collision_count)
                
                fitness = -score  # Negate because PSO minimizes (we want to maximize score)
            elif fitness_type == 'continuous':
                # Continuous fitness: total L2 error across all frames
                # Lower L2 error is better, PSO minimizes directly
                score = total_l2_error
                
                # BUG FIX: If total_l2_error is 0.0 but no valid L2 errors were accumulated,
                # this means all frames had NaN/Inf errors (model is broken), not perfect fitness!
                # Return a large penalty value instead of 0.0
                if score == 0.0 and valid_frame_count == 0:
                    fitness = float('inf')  # Worst possible fitness (will be rejected by PSO)
                else:
                    fitness = score  # Direct minimization of L2 error
            elif fitness_type == 'continuous2':
                # Continuous2 fitness: total L2 error + lambda * collision count
                lambda_col = 10.0
                
                if valid_frame_count > 0:
                    fitness = total_l2_error + lambda_col * collision_count
                else:
                    # No valid L2 errors: model is broken, use penalty fitness
                    fitness = float('inf')  # Worst possible fitness (will be rejected by PSO)
            else:
                raise ValueError(f"Unknown fitness_type: {fitness_type}")
        
        # Return fitness and frame counts for logging (only for original model evaluation)
        if use_original_l2_for_classification:
            frame_counts = {
                'positive_no_collision': positive_no_collision_count,
                'middle_no_collision': middle_no_collision_count,
                'negative_no_collision': negative_no_collision_count,
                'collision': collision_count,
                'total_evaluated': total_frames_evaluated
            }
            
            return float(fitness), frame_counts
        # Debug: print per-individual L2 and collision statistics
        self._fitness_eval_counter += 1
        mean_l2 = (total_l2_error / valid_frame_count) if valid_frame_count > 0 else float('inf')
        print(
            f"[fitness-debug] eval#{self._fitness_eval_counter} "
            f"total_l2={total_l2_error:.6f} mean_l2={mean_l2:.6f} "
            f"collision_count={collision_count} valid_frames={valid_frame_count}"
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
