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
import multiprocessing as mp
from mmcv.models.utils.functional import bivariate_gaussian_activation
from repair_common.arachne_base import ArachneBase

REPO_ROOT = Path(__file__).resolve().parents[3]

# Global collision metric instance (lazy initialization)
_COLLISION_METRIC = None

def _init_collision_worker():
    """Initialize collision metric in worker process."""
    global _COLLISION_METRIC
    if _COLLISION_METRIC is None:
        from mmcv.models.dense_heads.planning_head_plugin import UniADPlanningMetric
        _COLLISION_METRIC = UniADPlanningMetric()


def _compute_collision_from_occ(args):
    """
    Compute collision for UniAD using saved seg file.
    
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
        
        from mmcv.models.dense_heads.planning_head_plugin import UniADPlanningMetric
        metric = UniADPlanningMetric()
        seg_np = np.load(occ_path)['seg']  # UniAD uses 'seg' key, not 'occ'
        seg_t = torch.from_numpy(seg_np)
        if seg_t.dim() == 3:
            seg_t = seg_t.unsqueeze(0)
        pred_t = torch.from_numpy(pred_abs_np).unsqueeze(0)
        gt_t = torch.from_numpy(gt_abs_np).unsqueeze(0)
        _, obj_box_coll = metric.evaluate_coll(pred_t, gt_t, seg_t)
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
    
    # Ensure REPO_ROOT is available even in worker processes
    # (Global variables might not be reliable in some multiprocessing contexts)
    repo_root = Path(__file__).resolve().parents[3]

    path_str = str(occ_path)
    
    # Fix legacy path structure: insert /VAD/ if missing (UniAD might use similar structure?)
    if "baseline/vad_occ_cache" in path_str and "/VAD/" not in path_str:
        path_str = path_str.replace("baseline/vad_occ_cache", "baseline/VAD/vad_occ_cache")
    if "baseline/uniad_occ_cache" in path_str and "/UniAD/" not in path_str:
        path_str = path_str.replace("baseline/uniad_occ_cache", "baseline/UniAD/uniad_occ_cache")
    
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
    
    def set_options(self, **kwargs):
        """Set options for Arachne, including the UniAD trajectory decoding options."""
        super().set_options(**kwargs)
        if 'pred_traj_is_delta' in kwargs:
            self.pred_traj_is_delta = bool(kwargs['pred_traj_is_delta'])
        if 'apply_bivariate_activation' in kwargs:
            self.apply_bivariate_activation = bool(kwargs['apply_bivariate_activation'])
    
    def _get_collision_pool(self):
        """Lazily create a process pool for collision evaluation."""
        if self.collision_num_workers is None or self.collision_num_workers <= 1:
            return None
        if hasattr(self, '_collision_pool') and self._collision_pool is not None:
            if hasattr(self, '_collision_pool_workers') and self._collision_pool_workers == self.collision_num_workers:
                return self._collision_pool
        # Recreate pool if worker count changed
        if hasattr(self, '_collision_pool') and self._collision_pool is not None:
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
        all_gt_masks = []  # Store GT masks for L2 calculation
        all_original_predictions = []  # Store original predictions from JSON for comparison
        all_frame_data = []  # Store frame_data for later lookup
        
        # Collect all data (same as old version)
        for (token, scene_name), frame_data in frame_items:
            # Convert to numpy arrays (same as old version)
            ego_features = np.array(frame_data['ego_features'], dtype=np.float32)
            gt_trajectory = np.array(frame_data.get('gt_future_traj', []), dtype=np.float32)
            # Use ego_fut_cmd_idx to match old version
            cmd_idx = frame_data.get('ego_fut_cmd_idx', 0)
            # Get GT mask if available (for VAD rule L2 calculation)
            # VAD rule JSON uses 'ego_fut_masks' (plural), but we save as 'ego_fut_mask' (singular) in frame_data_dict
            gt_mask = frame_data.get('ego_fut_mask', frame_data.get('ego_fut_masks', None))
            if gt_mask is None:
                # If mask not available, assume all timesteps are valid (mask=1)
                # This matches VAD rule: if mask_sum > 0, frame is valid
                gt_mask = np.ones((6, 2), dtype=np.float32)  # Default: all valid
            else:
                gt_mask = np.array(gt_mask, dtype=np.float32)
                # Handle different mask shapes (matches convert_uniad_to_vad_metrics.py logic exactly)
                if gt_mask.ndim == 3:
                    gt_mask = gt_mask[0, :6, :2]  # Take first batch, 6 timesteps, 2 coords
                elif gt_mask.ndim == 2:
                    gt_mask = gt_mask[:6, :2] if gt_mask.shape[1] >= 2 else gt_mask[:6, None]
                elif gt_mask.ndim == 1:
                    gt_mask = gt_mask[:6, None]  # (6, 1) - will broadcast to (6, 2) in calculation
                else:
                    # Invalid shape, use default
                    gt_mask = np.ones((6, 2), dtype=np.float32)
                
                # Ensure final shape is (6, 2) for consistency
                if gt_mask.shape[0] < 6:
                    # Pad with 1.0 (valid)
                    padded = np.ones((6, 2), dtype=np.float32)
                    if gt_mask.ndim == 1:
                        padded[:gt_mask.shape[0], :] = gt_mask[:gt_mask.shape[0], None]
                    else:
                        min_dim = min(gt_mask.shape[1] if gt_mask.ndim > 1 else 1, 2)
                        padded[:gt_mask.shape[0], :min_dim] = gt_mask[:gt_mask.shape[0], :min_dim]
                    gt_mask = padded
                elif gt_mask.shape[0] >= 6:
                    # Slice to 6 timesteps
                    if gt_mask.ndim == 1:
                        # Broadcast (6,) to (6, 2)
                        gt_mask = np.column_stack([gt_mask[:6], gt_mask[:6]])
                    elif gt_mask.ndim == 2:
                        if gt_mask.shape[1] < 2:
                            # Broadcast (6, 1) to (6, 2)
                            gt_mask = np.column_stack([gt_mask[:6, 0], gt_mask[:6, 0]])
                        else:
                            gt_mask = gt_mask[:6, :2]
            all_ego_features.append(ego_features)
            all_gt_trajectories.append(gt_trajectory)
            all_cmd_indices.append(cmd_idx)
            all_has_collision.append(frame_data.get('has_collision', False))
            all_gt_masks.append(gt_mask)
            # Store original predictions from JSON for comparison (if available)
            original_pred = frame_data.get('original_predictions', None)
            all_original_predictions.append(original_pred)
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
                    batch_pred_raw = batch_pred_traj
                    
                    # Handle UniAD output shapes from repaired_layers
                    # UniAD reg_branch outputs: (batch, 12) = (batch, 6 timesteps × 2 coords)
                    if batch_pred_traj.dim() == 1:
                        batch_pred_traj = batch_pred_traj.unsqueeze(0)
                    elif batch_pred_traj.dim() == 2:
                        # UniAD format: (batch, 12) -> (batch, 6, 2)
                        if batch_pred_traj.shape[1] == 12:
                            batch_pred_traj = batch_pred_traj.view(-1, 6, 2)
                        else:
                            raise RuntimeError(
                                f"Unsupported pred_traj shape {tuple(batch_pred_traj.shape)}. "
                                f"Expected (batch, 12) for UniAD."
                            )
                    elif batch_pred_traj.dim() == 3:
                        # Already (batch, 6, 2) or (batch, timesteps, 2)
                        if batch_pred_traj.shape[1:] != (6, 2):
                            raise RuntimeError(
                                f"Unsupported pred_traj shape {tuple(batch_pred_traj.shape)}. "
                                f"Expected (batch, 6, 2) for UniAD."
                            )
                    
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

                    # Convert predictions to absolute trajectories for collision evaluation
                    batch_pred_abs = self._pred_traj_to_abs_gpu(batch_pred_raw, batch_cmd_tensor)
                    
                    # Always compute L2 for comparison/debugging, even if use_original_l2_for_classification=True
                    # For original model evaluation, we use JSON values for classification but still compute L2 for comparison
                    # For particle evaluation, compute L2 to reflect model changes
                    # Prepare GT masks tensor: (batch, 6, 2)
                    # gt_mask from all_gt_masks should already be (6, 2) after processing above
                    batch_gt_masks_tensor = torch.zeros(len(batch_gt_trajectories), 6, 2, dtype=torch.float32, device=device)
                    masks_loaded = 0
                    masks_default = 0
                    mask_sources = []  # Track where masks come from
                    for i, gt_mask in enumerate(all_gt_masks[batch_start:batch_end]):
                        # gt_mask should already be (6, 2) from processing above
                        if gt_mask.shape == (6, 2):
                            batch_gt_masks_tensor[i, :, :] = torch.tensor(gt_mask, dtype=torch.float32, device=device)
                            # Check if mask is all 1.0 (default) or has actual values
                            if np.allclose(gt_mask, 1.0):
                                masks_default += 1
                                # Check if mask was actually loaded from JSON or is default
                                frame_idx = batch_start + i
                                frame_data = all_frame_data[frame_idx] if frame_idx < len(all_frame_data) else None
                                mask_source = frame_data.get('_mask_field_used', 'unknown') if frame_data else 'unknown'
                                mask_sources.append(mask_source)
                            else:
                                masks_loaded += 1
                                mask_sources.append('loaded_from_json')
                        else:
                            # Fallback: handle unexpected shapes
                            min_len = min(gt_mask.shape[0], 6)
                            min_dim = min(gt_mask.shape[1], 2) if gt_mask.ndim > 1 else 2
                            if gt_mask.ndim == 1:
                                # Broadcast (6,) to (6, 2)
                                gt_mask_2d = np.column_stack([gt_mask[:6], gt_mask[:6]])
                                batch_gt_masks_tensor[i, :min_len, :] = torch.tensor(gt_mask_2d[:min_len, :], dtype=torch.float32, device=device)
                            else:
                                batch_gt_masks_tensor[i, :min_len, :min_dim] = torch.tensor(gt_mask[:min_len, :min_dim], dtype=torch.float32, device=device)
                            # Fill remaining with 1.0 (valid)
                            if min_len < 6:
                                batch_gt_masks_tensor[i, min_len:, :] = 1.0
                            if min_dim < 2:
                                batch_gt_masks_tensor[i, :, min_dim:] = 1.0
                            masks_default += 1
                            mask_sources.append('fallback_shape')
                        
                        
                        # Use _compute_l2_error_gpu which handles all shape cases correctly
                        # batch_pred_traj is delta format, _compute_l2_error_gpu will convert to absolute
                        batch_l2_errors = self._compute_l2_error_gpu(batch_pred_traj, batch_gt_tensor, batch_cmd_tensor, time_horizon=time_horizon, gt_mask=batch_gt_masks_tensor)
                        batch_l2_errors_np = batch_l2_errors.cpu().numpy()
                        
                                

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
                                # Fallback: if JSON value is missing, skip this frame
                                print(f"  [WARNING] Missing L2 value for frame {frame_idx}, skipping...")
                                continue
                        else:
                            # For particle evaluation, use computed L2 error so fitness reflects model changes
                            if batch_l2_errors_np is None:
                                raise RuntimeError("batch_l2_errors_np is None but use_original_l2_for_classification is False")
                            l2_error = float(batch_l2_errors_np[b_idx])
                        
                        # Note: frame_l2_comparison is populated in the accumulation section below
                        
                        # Check collision: use saved collision info from JSON if use_original_l2_for_classification
                        if use_original_l2_for_classification:
                            # For original model evaluation, use pre-computed collision from JSON for consistency
                            col_field = f'plan_obj_box_col_{time_horizon}s'
                            col_value = frame_data.get(col_field, 0.0)
                            has_collision = (col_value > 0)
                        elif collision_flags is not None:
                            has_collision = collision_flags[b_idx]
                        else:
                            # Recompute collision from occ/seg for particle evaluation
                            occ_path = _resolve_occ_path(frame_data.get('occ_path', None))
                            if not occ_path or not os.path.exists(occ_path):
                                raise ValueError(
                                    f"Missing occ_path for collision recomputation: {occ_path}. "
                                    "Please regenerate JSON with --occ-output-dir."
                                )
                            try:
                                if planning_metric is None:
                                    from mmcv.models.dense_heads.planning_head_plugin import UniADPlanningMetric
                                    planning_metric = UniADPlanningMetric().to(device)
                                seg_np = np.load(occ_path)['seg']
                                seg_t = torch.from_numpy(seg_np).to(device)
                                if seg_t.dim() == 3:
                                    seg_t = seg_t.unsqueeze(0)
                                pred_abs = batch_pred_abs[b_idx:b_idx + 1, :6, :2]
                                gt_abs = batch_gt_tensor[b_idx:b_idx + 1, :6, :2]
                                _, obj_box_col = planning_metric.evaluate_coll(pred_abs, gt_abs, seg_t)
                                if time_horizon == 1:
                                    col_value = float(obj_box_col[:2].mean().item())
                                elif time_horizon == 2:
                                    col_value = float(obj_box_col[:4].mean().item())
                                else:
                                    col_value = float(obj_box_col[:6].mean().item())
                                has_collision = (col_value > 0)
                            except Exception as e:
                                raise RuntimeError(f"Failed to compute collision from occ_path={occ_path}: {e}")
                        
                        # Track inf values
                        if np.isinf(l2_error) or l2_error == float('inf'):
                            inf_count += 1
                        
                        # Accumulate L2 error for continuous fitness
                        # Only accumulate valid L2 errors (not NaN or Inf)
                        # Note: We only process frames with fut_valid_flag=True, so all frames here are valid
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
                    print(f"  [ERROR] Batch processing failed: {e}", flush=True)
                    import traceback
                    traceback.print_exc()
                    continue
        
        
        # Compute fitness based on type (after all batches are processed)
        total_frames_evaluated = positive_no_collision_count + middle_no_collision_count + negative_no_collision_count + collision_count
        
        # Debug: print counts if all are zero
        if total_frames_evaluated == 0 and total_frames > 0:
            print(f"  [WARNING] No frames were evaluated! Total frames: {total_frames}, Error count: {error_count}", flush=True)
            if shape_errors:
                print(f"  [WARNING] Shape errors: {shape_errors[:5]}", flush=True)  # Print first 5 errors
            # If use_original_l2_for_classification and all frames failed, try to classify directly from JSON
            if use_original_l2_for_classification and error_count > 0:
                print(f"  [INFO] Attempting fallback: classifying frames directly from JSON data...", flush=True)
                # Fallback: classify directly from JSON without forward pass
                for (token, scene_name), frame_data in frame_items:
                    l2_field = f'plan_L2_{time_horizon}s'
                    col_field = f'plan_obj_box_col_{time_horizon}s'
                    original_l2 = frame_data.get(l2_field, None)
                    col_value = frame_data.get(col_field, 0.0)
                    has_collision = (col_value > 0)
                    
                    if original_l2 is None or np.isnan(original_l2) or np.isinf(original_l2):
                        continue
                    
                    l2_error = float(original_l2)
                    
                    if rep_method == 'semSegRep' and actual_threshold_good == actual_threshold_bad:
                        if l2_error < actual_threshold_good:
                            positive_no_collision_count += 1
                        else:
                            negative_no_collision_count += 1
                        if has_collision:
                            if l2_error < actual_threshold_good:
                                positive_no_collision_count -= 1
                            else:
                                negative_no_collision_count -= 1
                            collision_count += 1
                    else:
                        if has_collision:
                            collision_count += 1
                        else:
                            if l2_error < actual_threshold_good:
                                positive_no_collision_count += 1
                            elif l2_error <= actual_threshold_bad:
                                middle_no_collision_count += 1
                            else:
                                negative_no_collision_count += 1
                
                # Recalculate total after fallback
                total_frames_evaluated = positive_no_collision_count + middle_no_collision_count + negative_no_collision_count + collision_count
                print(f"  [INFO] Fallback classification complete: {total_frames_evaluated} frames evaluated", flush=True)
        
        # Compute fitness based on type (always compute, regardless of total_frames_evaluated)
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
            # Note: Only frames with fut_valid_flag=True are included in total_l2_error
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
            
        # Add debug logging similar to VAD (for individual evaluation)
        if not use_original_l2_for_classification:
            mean_l2 = total_l2_error / valid_frame_count if valid_frame_count > 0 else float('inf')
            print(f"[fitness-debug] fitness={fitness:.6f} total_l2={total_l2_error:.6f} mean_l2={mean_l2:.6f} "
                  f"collision_count={collision_count} valid_frames={valid_frame_count}", flush=True)

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
