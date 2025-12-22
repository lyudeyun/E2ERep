#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PyTorch implementation of Arachne algorithm for VAD repair"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no X server needed)
import matplotlib.pyplot as plt
import copy

# Import optimizers
# Handle both relative import (when used as package) and absolute import (when used as standalone module)
try:
    from .optimizers import get_optimizer
except ImportError:
    # When methods/arachne is added to sys.path and arachne_pytorch is imported directly,
    # use absolute import
    from optimizers import get_optimizer


class ArachnePyTorch:
    """PyTorch version of Arachne for neural network repair."""
    
    def __init__(self):
        """Initialize Arachne."""
        self.num_grad = None
        self.num_particles = 100
        self.num_iterations = 100
        self.num_input_pos_sampled = 200
        self.velocity_phi = 0.7  # Inertia weight (standard PSO: 0.4-0.9, commonly 0.7)
        self.min_iteration_range = 10
        self.target_layer = None
        self.num_weights_to_repair = None  # 新增：指定修复的权重数量
        self.early_stop_patience = None  # Early stopping patience (None = disabled, int = number of iterations without improvement)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_batch_size = None  # Batch size for fitness evaluation (None = auto-detect based on GPU memory)
        
        # Optimization algorithm selection
        self.optimization_algorithm = 'PSO'  # Options: 'PSO' or 'DE' (Differential Evolution)
        
        # Differential Evolution (DE) parameters
        self.de_crossover_rate = 0.9  # CR in [0, 1], typically 0.9
        self.de_mutation_factor = 0.8  # F in [0, 2], typically 0.5-1.0
        self.de_strategy = 'rand/1/bin'  # DE strategy: 'rand/1/bin', 'best/1/bin', 'rand/2/bin', etc.
    
    def set_options(self, **kwargs):
        """Set options for Arachne."""
        if 'num_grad' in kwargs:
            self.num_grad = kwargs['num_grad']
        if 'num_particles' in kwargs:
            self.num_particles = kwargs['num_particles']
        if 'num_iterations' in kwargs:
            self.num_iterations = kwargs['num_iterations']
        if 'num_input_pos_sampled' in kwargs:
            self.num_input_pos_sampled = kwargs['num_input_pos_sampled']
        if 'velocity_phi' in kwargs:
            self.velocity_phi = kwargs['velocity_phi']
        if 'min_iteration_range' in kwargs:
            self.min_iteration_range = kwargs['min_iteration_range']
        if 'early_stop_patience' in kwargs:
            self.early_stop_patience = kwargs['early_stop_patience']
        if 'target_layer' in kwargs:
            # Support single layer or list of layers
            target = kwargs['target_layer']
            if isinstance(target, (list, tuple)):
                self.target_layer = list(target)
            else:
                self.target_layer = [target] if target else None
        if 'target_layers' in kwargs:
            # Alternative parameter name
            self.target_layer = list(kwargs['target_layers'])
        if 'num_weights_to_repair' in kwargs:
            # 新增：指定要修复的权重数量
            self.num_weights_to_repair = kwargs['num_weights_to_repair']
        if 'eval_batch_size' in kwargs:
            # Batch size for fitness evaluation (None = auto-detect, int = fixed batch size)
            self.eval_batch_size = kwargs['eval_batch_size']
        if 'optimization_algorithm' in kwargs:
            # Optimization algorithm: 'PSO' or 'DE'
            algo = kwargs['optimization_algorithm'].upper()
            if algo not in ['PSO', 'DE']:
                raise ValueError(f"optimization_algorithm must be 'PSO' or 'DE', got '{algo}'")
            self.optimization_algorithm = algo
        if 'de_crossover_rate' in kwargs:
            # DE crossover rate (CR) in [0, 1]
            self.de_crossover_rate = kwargs['de_crossover_rate']
        if 'de_mutation_factor' in kwargs:
            # DE mutation factor (F) in [0, 2]
            self.de_mutation_factor = kwargs['de_mutation_factor']
        if 'de_strategy' in kwargs:
            # DE strategy: 'rand/1/bin', 'best/1/bin', 'rand/2/bin', etc.
            self.de_strategy = kwargs['de_strategy']
    
    def localize(self, model, input_neg, input_pos=None, output_dir=None, verbose=1, use_changed_unchanged=False):
        """
        Localize faulty neural weights using Gradient Loss (GL) and Forward Impact (FI).
        
        Note: GL and FI together form "Bidirectional Localization" (BL) - GL is backward,
        FI is forward. This method always uses BL (GL+FI).
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to be repaired
        input_neg : tuple (features, labels)
            Negative samples (features: [N, D], labels: [N])
        input_pos : tuple (features, labels), optional
            Positive samples (for Arachne v2 changed/unchanged method)
        output_dir : Path
            Directory to save localization results
        verbose : int
            Verbosity level
        use_changed_unchanged : bool
            If True, use Arachne v2 method: compute GL and FI separately for
            changed (negative) and unchanged (positive) samples, then combine
            using cost_chgd / (1 + cost_unchgd). This helps identify weights
            that affect negative samples more than positive samples.
            If False (default), use Arachne v1: compute GL and FI only on
            negative samples.
        
        Returns
        -------
        weights_to_repair : list
            List of [layer_name, i, j, GL, FI] for weights to repair
        """
        model.to(self.device)
        
        if self.num_grad is None:
            self.num_grad = len(input_neg[0]) * 20
        
        if use_changed_unchanged and input_pos is not None:
            # Arachne v2: Changed/Unchanged method
            # Compute GL and FI separately for changed and unchanged samples
            print("Using Arachne v2 Changed/Unchanged method...")
            print("  (Note: GL+FI together form Bidirectional Localization)")
            print("Step 1: Computing GL and FI for changed (negative) samples...")
            model.train()
            candidates_chgd = self._compute_gradient_loss(model, input_neg)
            model.eval()
            pool_chgd = self._compute_forward_impact(model, input_neg, candidates_chgd, self.num_grad)
            
            print("Step 2: Computing GL and FI for unchanged (positive) samples...")
            model.train()
            candidates_unchgd = self._compute_gradient_loss(model, input_pos)
            model.eval()
            pool_unchgd = self._compute_forward_impact(model, input_pos, candidates_unchgd, self.num_grad)
            
            print("Step 3: Combining costs (cost_chgd / (1 + cost_unchgd))...")
            weights_to_repair = self._extract_pareto_front_changed_unchanged(
                pool_chgd, pool_unchgd, output_dir
            )
        else:
            # Arachne v1: Standard localization (GL+FI on negative samples only)
            # Note: GL (backward) + FI (forward) = Bidirectional Localization (BL)
            print("Using Arachne v1 method (GL+FI on negative samples)...")
            print("  (Note: GL+FI together form Bidirectional Localization)")
            print("Step 1: Computing Gradient Loss (GL) - backward direction...")
            model.train()  # Need train mode for gradients
            candidates = self._compute_gradient_loss(model, input_neg)
            
            print("Step 2: Computing Forward Impact (FI) - forward direction...")
            model.eval()  # FI computation doesn't need gradients
            pool = self._compute_forward_impact(model, input_neg, candidates, self.num_grad)
            
            print("Step 3: Extracting Pareto Front from GL-FI space...")
            weights_to_repair = self._extract_pareto_front(pool, output_dir)
        
        if verbose:
            print(f"\nLocalization complete!")
            print(f"  Pareto-optimal weights: {len(weights_to_repair)}")
        
        return weights_to_repair
    
    def _compute_gradient_loss(self, model, input_neg):
        """
        Compute Gradient Loss for all weights in target layer(s).
        
        GL(w_ij) = |∂L/∂w_ij|
        
        IMPORTANT: This function uses the RAW decoder output (delta/displacement values),
        NOT absolute positions. This is correct for localization because GL/FI should
        be computed based on the model's raw behavior, not on post-processed values.
        """
        candidates = []
        
        # Convert to tensors (handle both tensor and numpy inputs)
        if isinstance(input_neg[0], torch.Tensor):
            X_neg = input_neg[0].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_neg = torch.tensor(input_neg[0], dtype=torch.float32).to(self.device)
        if isinstance(input_neg[1], torch.Tensor):
            y_neg = input_neg[1].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            y_neg = torch.tensor(input_neg[1], dtype=torch.float32).to(self.device)
        
        # Find target layers (can be multiple)
        target_layer_infos = self._find_target_layers(model)
        
        # Forward pass
        model.zero_grad()
        outputs = model(X_neg)
        
        # IMPORTANT: outputs here are RAW decoder outputs (delta/displacement values, not absolute positions)
        # This is correct for GL computation - we want to identify weights that contribute
        # to large raw predictions, which correlate with poor trajectory predictions.
        
        # For VAD trajectory prediction, we use L1Loss (same as VAD training)
        # The outputs should be [batch, 36] trajectory predictions (delta values)
        
        # For negative samples, we want to minimize the L1 magnitude of predictions
        # This helps identify weights that contribute to large predictions (bad cases)
        # We use L1 norm as proxy for prediction magnitude (consistent with VAD training)
        if outputs.dim() == 1:
            # If outputs is 1D [batch*36] or just [36], reshape appropriately
            if len(outputs) == 36:
                # Single sample compressed to [36], expand to [1, 36]
                outputs = outputs.unsqueeze(0)
            else:
                # Multiple samples compressed to [batch*36], reshape to [batch, 36]
                batch_size = len(outputs) // 36
                outputs = outputs.reshape(batch_size, 36)
        prediction_l1_magnitude = torch.sum(torch.abs(outputs), dim=1)  # [batch]
        
        # For negative samples, we want to minimize prediction L1 magnitude
        # So we compute loss = mean(prediction_l1_magnitude)
        loss = torch.mean(prediction_l1_magnitude)
        
        # Backward pass to get gradients
        loss.backward()
        
        # Collect gradients from all target layers
        for layer_name, layer in target_layer_infos:
            if layer.weight.grad is None:
                print(f"Warning: No gradient for {layer_name}, skipping")
                continue
            
            grad = layer.weight.grad.cpu().numpy()
            
            print(f"Processing layer: {layer_name}, shape: {grad.shape}")
            total_weights = grad.shape[0] * grad.shape[1]
            for j in range(grad.shape[0]):
                for i in range(grad.shape[1]):
                    gl = np.abs(grad[j, i])
                    candidates.append([layer_name, i, j, gl])
            print(f"  Completed processing {total_weights} weights for {layer_name}", flush=True)
        
        # Sort by gradient loss (descending)
        candidates.sort(key=lambda x: x[3], reverse=True)
        print(f"Total candidate weights from {len(target_layer_infos)} layer(s): {len(candidates)}")
        
        return candidates
    
    def _compute_forward_impact(self, model, input_neg, candidates, num_grad):
        """
        Compute Forward Impact for top-k candidates (supports multiple layers).
        
        FI(w_ij) = |activation_i × w_ij|
        
        IMPORTANT: This function uses raw layer activations (input to the target layer),
        NOT post-processed decoder outputs. This is correct for localization because FI
        measures the impact of weights on layer activations, independent of output format.
        """
        pool = {}
        _num_grad = min(num_grad, len(candidates))
        
        # Convert to tensor (handle both tensor and numpy inputs)
        if isinstance(input_neg[0], torch.Tensor):
            X_neg = input_neg[0].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_neg = torch.tensor(input_neg[0], dtype=torch.float32).to(self.device)
        
        # Pre-compute activations and weights for all unique layers in candidates
        layer_activations = {}
        layer_weights = {}
        
        unique_layers = set([c[0] for c in candidates[:_num_grad]])
        named_modules = dict(model.named_modules())
        
        for layer_name in unique_layers:
            target_layer = named_modules[layer_name]
            layer_activations[layer_name] = self._get_layer_activations(model, X_neg, layer_name)
            layer_weights[layer_name] = target_layer.weight.detach().cpu().numpy()
        
        # Compute FI for each candidate
        print(f"Computing FI for {_num_grad} candidates...", flush=True)
        for num in range(_num_grad):
            layer_name, i, j, gl = candidates[num]
            
            # FI = |activation_i × w_ji|
            activations = layer_activations[layer_name]
            weights = layer_weights[layer_name]
            
            activation_i = activations[0, i]  # Take first sample's activation
            w_ji = weights[j, i]
            fi = np.abs(activation_i * w_ji)
            
            pool[num] = [layer_name, i, j, gl, fi]
        
        print(f"  Completed computing FI for {_num_grad} candidates", flush=True)
        
        return pool
    
    def _get_layer_activations(self, model, input_data, target_layer_name):
        """Get activations from the layer before target layer."""
        activations = None
        
        def hook_fn(module, input, output):
            nonlocal activations
            activations = input[0].detach().cpu().numpy()
        
        # Register hook on target layer to capture its input
        target_layer = dict(model.named_modules())[target_layer_name]
        handle = target_layer.register_forward_hook(hook_fn)
        
        # Forward pass
        with torch.no_grad():
            _ = model(input_data)
        
        handle.remove()
        return activations
    
    def _find_target_layers(self, model):
        """Find target layers to repair (supports multiple layers)."""
        if self.target_layer is not None and len(self.target_layer) > 0:
            # Use specified layers
            target_layers = []
            named_modules = dict(model.named_modules())
            for layer_name in self.target_layer:
                if layer_name in named_modules:
                    target_layers.append((layer_name, named_modules[layer_name]))
                else:
                    print(f"Warning: Layer '{layer_name}' not found in model")
            
            if not target_layers:
                raise ValueError("None of the specified layers found in model")
            
            return target_layers
        
        # Find last Linear layer as default
        linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                linear_layers.append((name, module))
        
        if not linear_layers:
            raise ValueError("No Linear layer found in model")
        
        target_name, target_layer = linear_layers[-1]
        self.target_layer = [target_name]
        
        return [(target_name, target_layer)]
    
    def _extract_pareto_front(self, pool, output_dir=None):
        """
        Extract Pareto front from GL-FI objectives.
        
        If num_weights_to_repair is specified:
        - If needed > pareto_front: extract multi-layer pareto fronts
        - If needed < pareto_front: randomly sample from pareto front
        """
        # Collect objectives
        objectives = []
        indices = []
        
        for key in pool:
            weight = pool[key]
            gl = weight[3]  # Gradient Loss
            fi = weight[4]  # Forward Impact
            objectives.append([gl, fi])
            indices.append(key)
        
        scores = np.array(objectives)
        
        # Identify Pareto fronts (possibly multiple layers)
        if self.num_weights_to_repair is not None:
            pareto_indices = self._extract_pareto_with_target_count(
                scores, indices, self.num_weights_to_repair
            )
        else:
            # Original behavior: just extract first Pareto front
            pareto_mask = self._identify_pareto(scores)
            pareto_indices = np.where(pareto_mask)[0]
        
        # Save visualization
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Create mask for visualization
            pareto_mask = np.zeros(len(scores), dtype=bool)
            pareto_mask[pareto_indices] = True
            self._save_pareto_front(scores, pareto_mask, output_dir)
        
        # Extract Pareto-optimal weights
        weights_to_repair = []
        for idx in pareto_indices:
            key = indices[idx]
            weights_to_repair.append(pool[key])
        
        return weights_to_repair
    
    def _extract_pareto_front_changed_unchanged(self, pool_chgd, pool_unchgd, output_dir=None):
        """
        Extract Pareto front using Arachne v2 changed/unchanged method.
        
        This method combines costs from changed and unchanged samples:
        combined_cost = cost_chgd / (1 + cost_unchgd)
        
        This identifies weights that are:
        - Highly influential to changed (negative) behavior
        - Less influential to unchanged (positive) behavior
        
        Note: GL (backward) and FI (forward) together form "Bidirectional Localization" (BL).
        This method uses BL on both changed and unchanged samples separately.
        
        Parameters
        ----------
        pool_chgd : dict
            Pool of weights with GL and FI from changed (negative) samples
        pool_unchgd : dict
            Pool of weights with GL and FI from unchanged (positive) samples
        output_dir : Path, optional
            Directory to save visualization
        
        Returns
        -------
        weights_to_repair : list
            List of [layer_name, i, j, GL_chgd, FI_chgd] for weights to repair
        """
        # Match weights between changed and unchanged pools
        # Both pools should have the same structure (same candidates)
        matched_pool = {}
        
        for key in pool_chgd:
            if key in pool_unchgd:
                weight_chgd = pool_chgd[key]
                weight_unchgd = pool_unchgd[key]
                
                layer_name, i, j = weight_chgd[0], weight_chgd[1], weight_chgd[2]
                gl_chgd = weight_chgd[3]
                fi_chgd = weight_chgd[4]
                gl_unchgd = weight_unchgd[3]
                fi_unchgd = weight_unchgd[4]
                
                # Compute combined cost: cost_chgd / (1 + cost_unchgd)
                # Use GL and FI as 2D cost vector
                cost_chgd = np.array([gl_chgd, fi_chgd])
                cost_unchgd = np.array([gl_unchgd, fi_unchgd])
                
                # Arachne v2 formula: cost_chgd / (1 + cost_unchgd)
                # This gives higher score to weights with high chgd cost and low unchgd cost
                combined_cost = cost_chgd / (1.0 + cost_unchgd)
                
                matched_pool[key] = [
                    layer_name, i, j,
                    gl_chgd, fi_chgd,
                    gl_unchgd, fi_unchgd,
                    combined_cost
                ]
        
        # Collect combined costs for Pareto front extraction
        objectives = []
        indices = []
        
        for key in matched_pool:
            weight = matched_pool[key]
            combined_cost = weight[7]  # 2D array [GL_combined, FI_combined]
            objectives.append(combined_cost)
            indices.append(key)
        
        scores = np.array(objectives)
        
        # Extract Pareto front from combined costs
        if self.num_weights_to_repair is not None:
            pareto_indices = self._extract_pareto_with_target_count(
                scores, indices, self.num_weights_to_repair
            )
        else:
            pareto_mask = self._identify_pareto(scores)
            pareto_indices = np.where(pareto_mask)[0]
        
            # Save visualization
            if output_dir is not None:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                pareto_mask = np.zeros(len(scores), dtype=bool)
                pareto_mask[pareto_indices] = True
                self._save_pareto_front_changed_unchanged(scores, pareto_mask, output_dir, matched_pool)
        
        # Extract Pareto-optimal weights
        weights_to_repair = []
        for idx in pareto_indices:
            key = indices[idx]
            weight = matched_pool[key]
            # Return in format: [layer_name, i, j, GL_chgd, FI_chgd]
            # (compatible with existing code that expects 5 elements)
            weights_to_repair.append([
                weight[0], weight[1], weight[2],
                weight[3], weight[4]  # GL_chgd, FI_chgd
            ])
        
        print(f"Changed/unchanged method: {len(matched_pool)} matched weights, {len(weights_to_repair)} Pareto-optimal")
        
        return weights_to_repair
    
    def _save_pareto_front_changed_unchanged(self, scores, pareto_mask, output_dir, matched_pool):
        """Save changed/unchanged Pareto front visualization."""
        plt.figure(figsize=(12, 10))
        
        # Plot all points
        plt.scatter(scores[:, 0], scores[:, 1], c='blue', alpha=0.5, label='All Candidates')
        
        # Highlight Pareto front
        pareto_scores = scores[pareto_mask]
        plt.scatter(pareto_scores[:, 0], pareto_scores[:, 1], c='red', s=100,
                   marker='*', label='Pareto Front', zorder=10)
        
        plt.xlabel('Combined GL (GL_chgd / (1 + GL_unchgd))', fontsize=12)
        plt.ylabel('Combined FI (FI_chgd / (1 + FI_unchgd))', fontsize=12)
        plt.title('Arachne v2 Changed/Unchanged: Combined GL vs FI', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = output_dir / 'pareto_front_changed_unchanged.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Changed/unchanged Pareto front visualization saved to: {save_path}")
    
    def _extract_pareto_with_target_count(self, scores, indices, target_count):
        """
        Extract exactly target_count weights using Pareto fronts.
        
        Strategy:
        - If first Pareto front has >= target_count: randomly sample target_count
        - If first Pareto front has < target_count: add next Pareto layers
        
        Parameters
        ----------
        scores : np.ndarray
            GL-FI scores for all candidates
        indices : list
            Original indices in pool
        target_count : int
            Desired number of weights to repair
        
        Returns
        -------
        selected_indices : np.ndarray
            Indices of selected weights
        """
        n_points = len(scores)
        remaining_mask = np.ones(n_points, dtype=bool)  # Points not yet assigned to any front
        selected_indices = []
        
        layer = 1
        while len(selected_indices) < target_count and np.any(remaining_mask):
            # Extract next Pareto front from remaining points
            front_mask_local = self._identify_pareto(scores[remaining_mask])
            
            # Map back to global indices
            remaining_indices = np.where(remaining_mask)[0]
            front_indices = remaining_indices[front_mask_local]
            
            if len(selected_indices) + len(front_indices) <= target_count:
                # Add entire front
                selected_indices.extend(front_indices)
                remaining_mask[front_indices] = False
                print(f"  Layer {layer} Pareto front: {len(front_indices)} weights")
                layer += 1
            else:
                # Need to sample from this front
                needed = target_count - len(selected_indices)
                sampled = np.random.choice(front_indices, size=needed, replace=False)
                selected_indices.extend(sampled)
                print(f"  Layer {layer} Pareto front: sampled {needed}/{len(front_indices)} weights")
                break
        
        selected_indices = np.array(selected_indices)
        
        print(f"\nSelected {len(selected_indices)} weights (target: {target_count})")
        if len(selected_indices) < target_count:
            print(f"  Warning: Only found {len(selected_indices)} Pareto-optimal weights")
        
        return selected_indices
    
    def _identify_pareto(self, scores):
        """
        Identify Pareto front (maximizing both objectives).
        
        For maximization: point i dominates point j if:
        - scores[i] >= scores[j] in all objectives (GL and FI)
        - scores[i] > scores[j] in at least one objective
        
        Pareto front = points that are not dominated by any other point
        """
        n_points = scores.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        
        for i in range(n_points):
            if is_pareto[i]:
                # Find all points that i dominates
                # i dominates j if: scores[i] >= scores[j] in all dims AND scores[i] > scores[j] in at least one dim
                i_dominates = (
                    np.all(scores[i] >= scores, axis=1) &  # i >= others in all objectives
                    np.any(scores[i] > scores, axis=1)      # i > others in at least one objective
                )
                # Remove dominated points (but keep i itself)
                i_dominates[i] = False  # Don't remove i itself
                is_pareto[i_dominates] = False
        
        return is_pareto
    
    def _save_pareto_front(self, scores, pareto_mask, output_dir, filename='pareto_front.png'):
        """Save Pareto front visualization."""
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(scores[:, 0], scores[:, 1], c='blue', alpha=0.5, label='All Candidates')
        
        # Highlight Pareto front
        pareto_scores = scores[pareto_mask]
        plt.scatter(pareto_scores[:, 0], pareto_scores[:, 1], c='red', s=100, 
                   marker='*', label='Pareto Front', zorder=10)
        
        plt.xlabel('Gradient Loss (GL)', fontsize=12)
        plt.ylabel('Forward Impact (FI)', fontsize=12)
        plt.title('Pareto Front: GL vs FI', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = output_dir / filename
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Pareto front visualization saved to: {save_path}")
    
    def optimize(self, model, weights_to_repair, input_neg, input_pos, output_dir=None, verbose=1,
                 use_vad_eval=False, frame_data_dict=None, 
                 positive_frames=None, negative_frames=None, threshold_good=0.5, threshold_bad=1.0,
                 fitness_type='discrete', rep_method='Arachne_v1'):
        """
        Optimize weights using optimization algorithm (PSO or DE).
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to repair (wrapper model)
        weights_to_repair : list
            List of weights identified by localize()
        input_neg : tuple
            Negative samples (should be fixed)
        input_pos : tuple
            Positive samples (should remain correct)
        output_dir : Path
            Directory to save repaired model
        verbose : int
            Verbosity level
        use_vad_eval : bool
            If True, use open-loop VAD evaluation with saved ego_features
        frame_data_dict : dict
            Dictionary mapping (token, scene_name) to frame data
            Required if use_vad_eval=True. Each frame should contain:
            - 'ego_features': features before the repaired layers
            - 'gt_future_traj': ground truth trajectory
            - 'plan_L2_3s': original L2 error
        positive_frames : list
            List of (token, scene_name) for positive frames (required if use_vad_eval=True)
        negative_frames : list
            List of (token, scene_name) for negative frames (required if use_vad_eval=True)
        threshold_good : float
            L2 threshold for positive frames
        threshold_bad : float
            L2 threshold for negative frames
        middle_weight : float
            Weight for middle frames (default 0.5, meaning 2 middle = 1 positive)
        
        Returns
        -------
        repaired_model : nn.Module
            Repaired PyTorch model
        fitness_history : list
            List of best fitness values for each iteration
        """
        algo_name = self.optimization_algorithm
        print(f"\nStarting {algo_name} optimization...", flush=True)
        
        if use_vad_eval:
            if frame_data_dict is None:
                raise ValueError("use_vad_eval=True requires frame_data_dict")
            # For Arachne_v1: positive_frames and negative_frames are not used in evaluation
            # (evaluation uses ALL frames in frame_data_dict)
            # For semSegRep: they may be needed, so check if rep_method is semSegRep
            if rep_method != 'Arachne_v1' and (positive_frames is None or negative_frames is None):
                raise ValueError(f"use_vad_eval=True with rep_method={rep_method} requires positive_frames and negative_frames")
            if rep_method == 'Arachne_v1':
                print("Using open-loop VAD evaluation with saved ego_features", flush=True)
                print("  Note: positive_frames and negative_frames are not used (evaluates ALL frames)", flush=True)
            else:
                print("Using open-loop VAD evaluation with saved ego_features", flush=True)
        
        # Convert data to tensors (handle both tensor and numpy inputs)
        if isinstance(input_neg[0], torch.Tensor):
            X_neg = input_neg[0].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_neg = torch.tensor(input_neg[0], dtype=torch.float32).to(self.device)
        if isinstance(input_neg[1], torch.Tensor):
            y_neg = input_neg[1].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            y_neg = torch.tensor(input_neg[1], dtype=torch.float32).to(self.device)
        if isinstance(input_pos[0], torch.Tensor):
            X_pos = input_pos[0].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            X_pos = torch.tensor(input_pos[0], dtype=torch.float32).to(self.device)
        if isinstance(input_pos[1], torch.Tensor):
            y_pos = input_pos[1].clone().detach().to(dtype=torch.float32, device=self.device)
        else:
            y_pos = torch.tensor(input_pos[1], dtype=torch.float32).to(self.device)
        
        # Sample positive inputs if too many
        if len(X_pos) > self.num_input_pos_sampled:
            indices = np.random.choice(len(X_pos), self.num_input_pos_sampled, replace=False)
            X_pos = X_pos[indices]
            y_pos = y_pos[indices]
        
        # Print weight distribution across layers before initialization
        layer_counts = {}
        for weight_info in weights_to_repair:
            layer_name = weight_info[0]
            layer_counts[layer_name] = layer_counts.get(layer_name, 0) + 1
        print(f"  Weight distribution: {dict(layer_counts)}", flush=True)
        
        # Get optimizer class and create instance
        OptimizerClass = get_optimizer(algo_name)
        
        # Create optimizer instance with appropriate parameters
        if algo_name == 'PSO':
            optimizer = OptimizerClass(
                num_iterations=self.num_iterations,
                num_particles=self.num_particles,
                velocity_phi=self.velocity_phi,
                early_stop_patience=self.early_stop_patience,
                device=self.device,
                eval_batch_size=self.eval_batch_size
            )
        elif algo_name == 'DE':
            optimizer = OptimizerClass(
                num_iterations=self.num_iterations,
                num_particles=self.num_particles,
                de_crossover_rate=self.de_crossover_rate,
                de_mutation_factor=self.de_mutation_factor,
                de_strategy=self.de_strategy,
                early_stop_patience=self.early_stop_patience,
                device=self.device,
                eval_batch_size=self.eval_batch_size
            )
        else:
            # Generic optimizer (for future algorithms)
            optimizer = OptimizerClass(
                num_iterations=self.num_iterations,
                num_particles=self.num_particles,
                early_stop_patience=self.early_stop_patience,
                device=self.device,
                eval_batch_size=self.eval_batch_size
            )
        
        # Initialize population
        population = optimizer.initialize_population(model, weights_to_repair)
        if algo_name == 'PSO':
            print(f"Initialized {len(population)} particles", flush=True)
        elif algo_name == 'DE':
            print(f"Initialized {len(population)} individuals", flush=True)
        
        # Run optimization using strategy pattern
        # Pass fitness evaluation and weight application functions as callbacks
        best_model, fitness_history = optimizer.optimize(
            model, population, weights_to_repair,
            X_neg, y_neg, X_pos, y_pos,
            verbose=verbose,
            use_vad_eval=use_vad_eval,
            frame_data_dict=frame_data_dict,
            positive_frames=positive_frames,
            negative_frames=negative_frames,
            threshold_good=threshold_good,
            threshold_bad=threshold_bad,
            fitness_type=fitness_type,
            rep_method=rep_method,
            output_dir=output_dir,
            evaluate_fitness_fn=self._evaluate_fitness,
            evaluate_fitness_vad_fn=self._evaluate_fitness_vad_openloop,
            apply_weights_fn=self._apply_weight_adjustments
        )
        
        # Save repaired model and fitness history
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            save_path = output_dir / 'repaired_model.pth'
            torch.save(best_model.state_dict(), save_path)
            print(f"\nRepaired model saved to: {save_path}")
            
            # Save fitness history
            history_path = output_dir / 'fitness_history.json'
            with open(history_path, 'w') as f:
                json.dump({
                    'iterations': list(range(0, len(fitness_history))),  # Start from 0 (original)
                    'best_fitness': fitness_history,
                    'original_fitness': fitness_history[0],
                    'final_fitness': fitness_history[-1],
                    'improvement': fitness_history[0] - fitness_history[-1]
                }, f, indent=2)
            print(f"Fitness history saved to: {history_path}")
        
        return best_model, fitness_history
    
    def _apply_weight_adjustments(self, model, weights_to_repair, position):
        """Apply weight adjustments to model with safety bounds."""
        modified_model = copy.deepcopy(model)
        
        # Get named modules for debugging
        named_modules = dict(modified_model.named_modules())
        
        # Pre-compute original weight ranges for each layer (for safety bounds)
        layer_weight_bounds = {}
        for weight_info in weights_to_repair:
            layer_name = weight_info[0]
            if layer_name not in layer_weight_bounds:
                if layer_name in named_modules:
                    layer = named_modules[layer_name]
                    layer_weights = layer.weight.detach().cpu().numpy()
                    weight_min = float(np.min(layer_weights))  # lb
                    weight_max = float(np.max(layer_weights))  # ub
                    weight_range = weight_max - weight_min
                    # Allow weights to vary symmetrically: extend range by 0.5x on each side
                    # This works correctly even when lb is negative
                    # New range: [lb - 0.5*range, ub + 0.5*range]
                    # This expands the range symmetrically from both ends (more conservative)
                    layer_weight_bounds[layer_name] = {
                        'min': weight_min - weight_range * 0.5,
                        'max': weight_max + weight_range * 0.5
                    }
                else:
                    # Fallback: reasonable default bounds
                    layer_weight_bounds[layer_name] = {'min': -10.0, 'max': 10.0}
        
        for weight_info in weights_to_repair:
            layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
            key = (layer_name, i, j)
            
            if key in position:
                # Get the layer - check if layer_name exists in named_modules
                if layer_name not in named_modules:
                    # Try to find it by checking all module names
                    found = False
                    for full_name, module in named_modules.items():
                        if full_name.endswith('.' + layer_name) or full_name == layer_name:
                            layer = module
                            found = True
                            break
                    if not found:
                        # Debug: print available layer names
                        available_layers = [name for name in named_modules.keys() if 'layer' in name.lower()]
                        print(f"Warning: Layer '{layer_name}' not found in named_modules!")
                        print(f"  Available layers containing 'layer': {available_layers[:10]}")
                        continue
                else:
                    layer = named_modules[layer_name]
                
                # Direct weight replacement (not adjustment)
                with torch.no_grad():
                    # PyTorch weight shape: [out_features, in_features]
                    new_weight = position[key]  # Directly use position as weight value
                    
                    # Clamp weight to safe bounds to prevent NaN/Inf
                    bounds = layer_weight_bounds[layer_name]
                    new_weight_clamped = np.clip(new_weight, bounds['min'], bounds['max'])
                    
                    # Only apply if weight is valid
                    if np.isfinite(new_weight_clamped):
                        layer.weight[j, i] = new_weight_clamped
                    else:
                        # If somehow still invalid, use original weight
                        original_weight = layer.weight[j, i].item()
                        layer.weight[j, i] = original_weight
        
        return modified_model
    
    def _evaluate_fitness(self, model, X_neg, y_neg, X_pos, y_pos, loss_fn):
        """
        Evaluate fitness function.
        
        Fitness = loss_neg + lambda * loss_pos
        (Want to minimize negative loss while not increasing positive loss)
        """
        model.eval()
        
        with torch.no_grad():
            # Negative samples: should be fixed
            outputs_neg = model(X_neg)
            if outputs_neg.dim() == 1:
                outputs_neg = outputs_neg.unsqueeze(1)
            loss_neg = loss_fn(outputs_neg, y_neg.unsqueeze(1) if y_neg.dim() == 1 else y_neg)
            
            # Positive samples: should remain correct
            outputs_pos = model(X_pos)
            if outputs_pos.dim() == 1:
                outputs_pos = outputs_pos.unsqueeze(1)
            loss_pos = loss_fn(outputs_pos, y_pos.unsqueeze(1) if y_pos.dim() == 1 else y_pos)
            
            # Combined fitness (weighted sum)
            lambda_pos = 1.0  # Weight for positive loss
            fitness = loss_neg.item() + lambda_pos * loss_pos.item()
        
        return fitness
    
    def _evaluate_fitness_vad_openloop(self, repaired_layers, frame_data_dict, 
                                        positive_frames, negative_frames,
                                        threshold_good, threshold_bad, fitness_type='discrete',
                                        rep_method='Arachne_v1', use_original_l2_for_classification=False):
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
            fitness = mean L2 error + lambda * N_col
            where:
                mean L2 error: average L2 error across all valid frames
                N_col: number of frames with collision
                lambda: penalty coefficient (default 10.0)
        
        For continuous3 fitness:
            fitness = total L2 error + lambda * N_col
            (uses the same collision penalty as continuous2 but keeps the total L2 sum)
        
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
            Type of fitness function ('discrete', 'continuous', 'continuous2', 'continuous3')
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
            # Calculate median L2 error from original JSON data (plan_L2_3s)
            all_original_l2 = []
            for frame_data in frame_data_dict.values():
                original_l2 = frame_data.get('plan_L2_3s', None)
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
        #   - For Arachne_v1: positive_frames/negative_frames are not used (evaluates ALL frames), so disable fast path
        #   - For semSegRep with median threshold: cannot use fast path because positive_frames/negative_frames
        #     were extracted using a different threshold (0.5) than what we use for evaluation (median L2)
        can_use_fast_path = (use_original_l2_for_classification and 
                            rep_method != 'Arachne_v1' and  # Arachne_v1 doesn't use positive_frames/negative_frames
                            positive_frames is not None and negative_frames is not None and 
                            len(positive_frames) > 0 and len(negative_frames) > 0 and
                            not (rep_method == 'semSegRep' and use_median_threshold))
        
        if use_original_l2_for_classification:
            if rep_method == 'Arachne_v1':
                # Arachne_v1 doesn't use positive_frames/negative_frames, always use normal evaluation
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
        
        with torch.no_grad():
            # OPTIMIZED: Batch processing for GPU acceleration
            # Determine batch size: use configured value or auto-detect based on GPU memory
            if self.eval_batch_size is not None:
                batch_size = self.eval_batch_size
            else:
                # Auto-detect batch size based on GPU memory
                if device.type == 'cuda':
                    # Get GPU memory info
                    gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                    # Conservative estimate: ~50MB per batch (for features + predictions + intermediates)
                    # For 32GB GPU: ~640, but use 256 as safe default
                    # For 16GB GPU: ~320, but use 128 as safe default
                    # For 8GB GPU: ~160, but use 64 as safe default
                    if gpu_memory_gb >= 24:
                        batch_size = 256  # Large GPU (V100, A100, etc.)
                    elif gpu_memory_gb >= 12:
                        batch_size = 128  # Medium GPU (RTX 3090, etc.)
                    else:
                        batch_size = 64   # Small GPU or default (RTX 3080, T4, etc.)
                else:
                    batch_size = 32  # CPU: smaller batch size
            
            frame_items = list(frame_data_dict.items())
            total_frames = len(frame_items)
            
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
                    
                    # Prepare GT trajectories tensor: (batch, 6, 2) (same as old version)
                    batch_gt_tensor = torch.zeros(len(batch_gt_trajectories), 6, 2, dtype=torch.float32, device=device)
                    for i, gt_traj in enumerate(batch_gt_trajectories):
                        if len(gt_traj.shape) == 1:
                            gt_traj = gt_traj.reshape(-1, 2)
                        min_len = min(gt_traj.shape[0], 6)
                        batch_gt_tensor[i, :min_len, :] = torch.tensor(gt_traj[:min_len], dtype=torch.float32, device=device)
                    
                    # Prepare cmd_indices tensor: (batch,)
                    batch_cmd_tensor = torch.tensor(batch_cmd_indices, dtype=torch.long, device=device)
                    
                    # Use _compute_l2_error_gpu which handles all shape cases correctly (same as old version)
                    batch_l2_errors = self._compute_l2_error_gpu(batch_pred_traj, batch_gt_tensor, batch_cmd_tensor)
                    batch_l2_errors_np = batch_l2_errors.cpu().numpy()
                    
                    # Process each frame in batch for classification and statistics (same as old version)
                    for b_idx in range(batch_end - batch_start):
                        frame_idx = batch_start + b_idx
                        frame_data = all_frame_data[frame_idx]
                        
                        # For original model evaluation, use pre-computed L2 from JSON for consistency
                        if use_original_l2_for_classification:
                            original_l2 = frame_data.get('plan_L2_3s', None)
                            if original_l2 is not None and not np.isnan(original_l2) and not np.isinf(original_l2):
                                l2_error = float(original_l2)
                            else:
                                # Fallback to computed L2 if plan_L2_3s is missing
                                l2_error = float(batch_l2_errors_np[b_idx])
                        else:
                            # For particle evaluation, use computed L2 error so fitness reflects model changes
                            l2_error = float(batch_l2_errors_np[b_idx])
                        
                        has_collision = batch_has_collision[b_idx]  # Python bool, not tensor
                        
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
                            # For original model: uses plan_L2_3s from JSON (consistent with baseline)
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
            
            # Compute fitness based on type (same as old version)
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
                # Continuous2 fitness: mean L2 error + lambda * collision count (applies to all methods)
                lambda_col = 10.0
                
                if valid_frame_count > 0:
                    mean_l2_error = total_l2_error / valid_frame_count
                    fitness = mean_l2_error + lambda_col * collision_count
                else:
                    # No valid L2 errors: model is broken, use penalty fitness
                    fitness = float('inf')  # Worst possible fitness (will be rejected by PSO)
            elif fitness_type == 'continuous3':
                # Continuous3 fitness: total L2 error + lambda * collision count
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
    
    def _compute_l2_error_gpu(self, pred_traj, gt_traj, cmd_idx):
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
        
        Returns
        -------
        l2_errors : torch.Tensor
            L2 errors for each sample, shape: (batch,)
        """
        batch_size = pred_traj.shape[0]
        # B2D fixed planning horizon: 6 future steps
        timesteps = 6
        
        # Accept B2D flattened output: (batch, 72) -> (batch, 6, 6, 2) -> select cmd -> (batch, 6, 2)
        if pred_traj.dim() == 2:
            if pred_traj.shape[-1] != 72:
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 72) for B2D."
                )
            pred_traj = pred_traj.view(batch_size, 6, timesteps, 2)
            if isinstance(cmd_idx, torch.Tensor):
                cmd_idx_clamped = torch.clamp(cmd_idx.long(), 0, 5)
                pred_traj = pred_traj[torch.arange(batch_size, device=pred_traj.device), cmd_idx_clamped]
            else:
                cmd_idx_clamped = max(0, min(5, int(cmd_idx)))
                pred_traj = pred_traj[:, cmd_idx_clamped]
        elif pred_traj.dim() == 4:
            # Explicit: (batch, 6, 6, 2)
            if pred_traj.shape[1:] != (6, timesteps, 2):
                raise RuntimeError(
                    f"Unsupported pred_traj shape {tuple(pred_traj.shape)}. "
                    f"Expected (batch, 6, 6, 2) for B2D."
                )
            if isinstance(cmd_idx, torch.Tensor):
                cmd_idx_clamped = torch.clamp(cmd_idx.long(), 0, 5)
                pred_traj = pred_traj[torch.arange(batch_size, device=pred_traj.device), cmd_idx_clamped]
            else:
                cmd_idx_clamped = max(0, min(5, int(cmd_idx)))
                pred_traj = pred_traj[:, cmd_idx_clamped]
        elif pred_traj.dim() == 3:
            # Already selected trajectory: (batch, 6, 2)
            if pred_traj.shape[1:] != (timesteps, 2):
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
        
        # Ensure gt_traj has correct shape: (batch, timesteps, 2)
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
        
        # Handle shape mismatches
        min_len = min(pred_traj_abs.shape[1], gt_traj.shape[1])
        if min_len == 0:
            return torch.full((batch_size,), float('inf'), device=pred_traj_abs.device, dtype=pred_traj_abs.dtype)
        pred_traj_abs = pred_traj_abs[:, :min_len, :]
        gt_traj = gt_traj[:, :min_len, :]
        
        # Compute L2 distance for each timestep: (batch, timesteps)
        diff = pred_traj_abs - gt_traj
        l2_distances = torch.sqrt(torch.sum(diff**2, dim=2))  # (batch, timesteps)
        
        # Return average L2 error over all timesteps: (batch,)
        return torch.mean(l2_distances, dim=1)
    
    def compute_l2_statistics(self, model, frame_data_dict, threshold_good=0.5, threshold_bad=2.0):
        """
        Compute L2 error statistics for all frames.
        
        Uses batch processing for GPU acceleration (similar to _evaluate_fitness_vad_openloop).
        
        Parameters
        ----------
        model : nn.Module
            Model to evaluate
        frame_data_dict : dict
            Dictionary mapping (token, scene_name) to frame data
        threshold_good : float
            L2 threshold for good frames
        threshold_bad : float
            L2 threshold for bad frames
        
        Returns
        -------
        stats : dict
            Statistics dictionary
        """
        model.eval()
        device = next(model.parameters()).device
        
        # Determine batch size based on GPU memory
        if self.eval_batch_size is not None:
            batch_size = self.eval_batch_size
        else:
            if device.type == 'cuda':
                gpu_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                if gpu_memory_gb >= 24:
                    batch_size = 256
                elif gpu_memory_gb >= 12:
                    batch_size = 128
                else:
                    batch_size = 64
            else:
                batch_size = 32
        
        frame_items = list(frame_data_dict.items())
        total_frames = len(frame_items)
        
        # Pre-allocate lists for batch processing
        all_ego_features = []
        all_gt_trajectories = []
        all_cmd_indices = []
        all_has_collision = []
        
        # Collect all data
        for (token, scene_name), frame_data in frame_items:
            all_ego_features.append(frame_data['ego_features'])
            all_gt_trajectories.append(frame_data['gt_future_traj'])
            # Use 'ego_fut_cmd_idx' to match old version and build_frame_data_dict
            all_cmd_indices.append(frame_data.get('ego_fut_cmd_idx', frame_data.get('cmd_idx', 0)))
            all_has_collision.append(frame_data.get('has_collision', False))
        
        # Convert to tensors (torch.stack handles numpy arrays automatically)
        all_ego_features = torch.stack([torch.tensor(feat, dtype=torch.float32) if not isinstance(feat, torch.Tensor) else feat for feat in all_ego_features]).to(device)
        all_gt_trajectories = torch.stack([torch.tensor(traj, dtype=torch.float32) for traj in all_gt_trajectories]).to(device)
        all_cmd_indices = torch.tensor(all_cmd_indices, dtype=torch.long).to(device)
        all_has_collision = torch.tensor(all_has_collision, dtype=torch.bool).to(device)
        
        # Batch process
        all_l2_errors = []
        collision_count = 0
        
        with torch.no_grad():
            for batch_start in range(0, total_frames, batch_size):
                batch_end = min(batch_start + batch_size, total_frames)
                batch_ego_features = all_ego_features[batch_start:batch_end]
                batch_gt_trajectories = all_gt_trajectories[batch_start:batch_end]
                batch_cmd_indices = all_cmd_indices[batch_start:batch_end]
                batch_has_collision = all_has_collision[batch_start:batch_end]
                
                # Forward pass
                batch_pred_traj = model(batch_ego_features)
                
                # Ensure batch_gt_trajectories has correct shape: (batch, 6, 2)
                # Each gt_trajectory should be (6, 2), so after stacking it should be (batch, 6, 2)
                if batch_gt_trajectories.dim() == 2:
                    # If somehow it's (batch*6, 2), reshape it
                    if batch_gt_trajectories.shape[0] == batch_end - batch_start * 6:
                        batch_gt_trajectories = batch_gt_trajectories.view(batch_end - batch_start, 6, 2)
                    else:
                        # Single trajectory: expand to batch
                        batch_size_local = batch_end - batch_start
                        if batch_gt_trajectories.shape == (6, 2):
                            batch_gt_trajectories = batch_gt_trajectories.unsqueeze(0).expand(batch_size_local, -1, -1)
                        else:
                            # Reshape and pad if needed
                            batch_gt_trajectories = batch_gt_trajectories.view(-1, 2).unsqueeze(0)
                            if batch_gt_trajectories.shape[1] != 6:
                                padded_gt = torch.zeros(1, 6, 2, device=batch_gt_trajectories.device, dtype=batch_gt_trajectories.dtype)
                                min_timesteps = min(batch_gt_trajectories.shape[1], 6)
                                padded_gt[:, :min_timesteps, :] = batch_gt_trajectories[:, :min_timesteps, :]
                                batch_gt_trajectories = padded_gt.expand(batch_size_local, -1, -1)
                
                # Use _compute_l2_error_gpu which handles all shape cases correctly
                batch_l2_errors = self._compute_l2_error_gpu(batch_pred_traj, batch_gt_trajectories, batch_cmd_indices)
                batch_l2_errors_np = batch_l2_errors.cpu().numpy()
                
                # Collect results
                for b_idx in range(batch_end - batch_start):
                    all_l2_errors.append(float(batch_l2_errors_np[b_idx]))
                    if batch_has_collision[b_idx].item():
                        collision_count += 1
        
        all_l2_errors = np.array(all_l2_errors)
        
        # Calculate frame counts (matching old version format)
        positive_count = int(np.sum(all_l2_errors < threshold_good))
        middle_count = int(np.sum((all_l2_errors >= threshold_good) & (all_l2_errors <= threshold_bad)))
        negative_count = int(np.sum(all_l2_errors > threshold_bad))
        
        # Return stats matching old version format (for compatibility with repair_ego_fut_decoder_arachne.py)
        stats = {
            'total_l2': float(np.sum(all_l2_errors)),  # Total L2 error (for compatibility)
            'mean_l2': float(np.mean(all_l2_errors)),
            'median_l2': float(np.median(all_l2_errors)),
            'std_l2': float(np.std(all_l2_errors)),
            'min_l2': float(np.min(all_l2_errors)),
            'max_l2': float(np.max(all_l2_errors)),
            'positive_count': positive_count,  # L2 < threshold_good
            'middle_count': middle_count,  # threshold_good <= L2 <= threshold_bad
            'negative_count': negative_count,  # L2 > threshold_bad
            'total_frames': len(all_l2_errors),
            'collision_frames': collision_count,
            'all_l2_errors': all_l2_errors.tolist()  # For compatibility with old version
        }
        
        return stats


def load_repair_data(json_file, threshold_good=0.3, threshold_bad=1.0, l2_metric='plan_L2_3s'):
    """
    Load repair data from VAD JSON file.
    
    Returns
    -------
    input_neg, input_pos : tuple of (features, labels)
    """
    print(f"Loading data from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    positive_features = []
    negative_features = []
    
    for frame in data:
        if not frame.get('fut_valid_flag', False):
            continue
        
        if 'ego_features' not in frame:
            continue
        
        l2_error = frame.get(l2_metric, None)
        if l2_error is None:
            continue
        
        ego_features = np.array(frame['ego_features'])
        
        if l2_error < threshold_good:
            positive_features.append(ego_features)
        elif l2_error > threshold_bad:
            negative_features.append(ego_features)
    
    print(f"Loaded {len(positive_features)} positive samples (L2 < {threshold_good})")
    print(f"Loaded {len(negative_features)} negative samples (L2 > {threshold_bad})")
    
    # Convert to numpy arrays
    X_pos = np.array(positive_features, dtype=np.float32)
    y_pos = np.ones(len(positive_features), dtype=np.float32)
    
    X_neg = np.array(negative_features, dtype=np.float32)
    y_neg = np.zeros(len(negative_features), dtype=np.float32)
    
    input_pos = (X_pos, y_pos)
    input_neg = (X_neg, y_neg)
    
    return input_neg, input_pos


def extract_frame_identifiers(json_file, threshold_good=0.3, threshold_bad=1.0, 
                               l2_metric='plan_L2_3s', max_positive=None, max_negative=None):
    """
    Extract frame identifiers for positive and negative samples.
    
    Parameters
    ----------
    json_file : str
        Path to VAD evaluation JSON
    threshold_good : float
        L2 threshold for positive frames
    threshold_bad : float
        L2 threshold for negative frames
    l2_metric : str
        L2 metric name in JSON
    max_positive : int, optional
        Maximum number of positive frames to return
    max_negative : int, optional
        Maximum number of negative frames to return
    
    Returns
    -------
    positive_frames : list of (batch_idx, frame_idx) or (token, scene_name)
    negative_frames : list of (batch_idx, frame_idx) or (token, scene_name)
    """
    print(f"Extracting frame identifiers from {json_file}...")
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    positive_frames = []
    negative_frames = []
    
    for idx, frame in enumerate(data):
        if not frame.get('fut_valid_flag', False):
            continue
        
        l2_error = frame.get(l2_metric, None)
        if l2_error is None:
            continue
        
        # Try to get identifiers (support both formats)
        token = frame.get('token')
        scene_name = frame.get('scene_name')
        batch_idx = frame.get('batch_idx')
        frame_idx = frame.get('frame_idx')
        
        # Use whichever identifier is available
        if token and scene_name:
            frame_id = (token, scene_name)
        elif batch_idx is not None and frame_idx is not None:
            frame_id = (batch_idx, frame_idx)
        else:
            # Fallback: use index in the JSON array
            frame_id = (idx, 0)
        
        if l2_error < threshold_good:
            positive_frames.append(frame_id)
        elif l2_error > threshold_bad:
            negative_frames.append(frame_id)
    
    # Sample if needed
    if max_positive and len(positive_frames) > max_positive:
        indices = np.random.choice(len(positive_frames), max_positive, replace=False)
        positive_frames = [positive_frames[i] for i in indices]
    
    if max_negative and len(negative_frames) > max_negative:
        indices = np.random.choice(len(negative_frames), max_negative, replace=False)
        negative_frames = [negative_frames[i] for i in indices]
    
    print(f"Extracted {len(positive_frames)} positive frames (L2 < {threshold_good})")
    print(f"Extracted {len(negative_frames)} negative frames (L2 > {threshold_bad})")
    
    return positive_frames, negative_frames


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
        frame_data = {
            'ego_features': np.array(frame['ego_features'], dtype=np.float32),
            'plan_L2_3s': frame.get('plan_L2_3s', 0.0),
            'ego_fut_cmd_idx': frame.get('ego_fut_cmd_idx', 0),  # 保存指令索引
        }
        
        # Add collision information if available
        col_1s = frame.get('plan_obj_box_col_1s', 0.0)
        col_2s = frame.get('plan_obj_box_col_2s', 0.0)
        col_3s = frame.get('plan_obj_box_col_3s', 0.0)
        frame_data['has_collision'] = (col_1s > 0) or (col_2s > 0) or (col_3s > 0)
        frame_data['plan_obj_box_col_1s'] = float(col_1s) if isinstance(col_1s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_2s'] = float(col_2s) if isinstance(col_2s, (int, float)) else 0.0
        frame_data['plan_obj_box_col_3s'] = float(col_3s) if isinstance(col_3s, (int, float)) else 0.0
        
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
    from arachne_pytorch import ArachnePyTorch, load_repair_data
    
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
