"""
Particle Swarm Optimization (PSO) optimizer for neural network repair.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import copy


class PSOOptimizer:
    """
    Particle Swarm Optimization optimizer.
    
    PSO uses particles that move through the search space with velocities
    influenced by their own best position and the global best position.
    """
    
    def __init__(self, num_iterations=100, num_particles=100, 
                 velocity_phi=0.7, early_stop_patience=None, 
                 device=None, eval_batch_size=None):
        """
        Initialize PSO optimizer.
        
        Parameters
        ----------
        num_iterations : int
            Maximum number of iterations
        num_particles : int
            Number of particles in the swarm
        velocity_phi : float
            Inertia weight (typically 0.4-0.9, commonly 0.7)
        early_stop_patience : int or None
            Early stopping patience
        device : torch.device
            Device to run optimization on
        eval_batch_size : int or None
            Batch size for fitness evaluation
        """
        self.num_iterations = num_iterations
        self.num_particles = num_particles
        self.velocity_phi = velocity_phi
        self.early_stop_patience = early_stop_patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_batch_size = eval_batch_size
    
    def initialize_population(self, model, weights_to_repair):
        """
        Initialize PSO particles using original weight values with small perturbations.
        
        For each weight to repair, initialize near its original value with small random perturbation.
        This ensures initial population is close to the original model, providing better starting point.
        Same method as old PSO implementation.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model
        weights_to_repair : list
            List of weights to repair [layer_name, i, j, GL, FI]
        
        Returns
        -------
        particles : list
            List of particle dictionaries
        """
        # Pre-compute weight ranges for each layer (for bounds and initialization)
        # Same structure as old PSO: compute extended bounds and store both original and extended ranges
        named_modules = dict(model.named_modules())
        layer_ranges = {}  # {layer_name: {'min': float, 'max': float, 'range': float, 'original_min': float, 'original_max': float, 'original_range': float}}
        
        for weight_info in weights_to_repair:
            layer_name = weight_info[0]
            if layer_name not in layer_ranges:
                if layer_name in named_modules:
                    layer = named_modules[layer_name]
                    layer_weights = layer.weight.detach().cpu().numpy()
                    weight_min = float(np.min(layer_weights))
                    weight_max = float(np.max(layer_weights))
                    weight_range = weight_max - weight_min
                    # Store extended bounds (same as old PSO)
                    layer_ranges[layer_name] = {
                        'min': weight_min - weight_range * 0.5,  # Extended bound
                        'max': weight_max + weight_range * 0.5,  # Extended bound
                        'range': weight_range,  # Original range
                        'original_min': weight_min,
                        'original_max': weight_max,
                        'original_range': weight_range
                    }
                else:
                    # Fallback
                    layer_ranges[layer_name] = {'min': -1.0, 'max': 1.0, 'range': 2.0, 'original_min': -1.0, 'original_max': 1.0, 'original_range': 2.0}
        
        # Initialize particles
        particles = []
        for _ in range(self.num_particles):
            particle = {
                'position': {},
                'velocity': {},
                'best_position': {},
                'best_fitness': float('inf')
            }
            
            # Initialize near original weights with small perturbations (same as old PSO)
            for weight_info in weights_to_repair:
                layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                key = (layer_name, i, j)
                
                # Get original weight value
                if layer_name in named_modules:
                    layer = named_modules[layer_name]
                    original_weight = float(layer.weight[j, i].item())
                else:
                    layer_info = layer_ranges.get(layer_name, {'min': -1.0, 'max': 1.0, 'original_min': -1.0, 'original_max': 1.0})
                    original_weight = (layer_info.get('original_min', -1.0) + layer_info.get('original_max', 1.0)) / 2.0
                
                # Initialize near original with small perturbation
                layer_info = layer_ranges.get(layer_name, {'min': -1.0, 'max': 1.0, 'range': 2.0})
                weight_min = layer_info['min']
                weight_max = layer_info['max']
                weight_range = layer_info.get('range', weight_max - weight_min)
                
                perturbation_range = weight_range * 0.1  # 10% of weight range
                initial_weight = original_weight + np.random.uniform(-perturbation_range, perturbation_range)
                initial_weight = np.clip(initial_weight, weight_min, weight_max)
                particle['position'][key] = float(initial_weight)
                
                # Initialize velocity (small random value, same as old PSO)
                velocity_range = weight_range * 0.1  # 10% of weight range
                particle['velocity'][key] = np.random.uniform(-velocity_range, velocity_range)
                particle['best_position'][key] = particle['position'][key]
            
            particles.append(particle)
        
        return particles
    
    def optimize(self, model, particles, weights_to_repair,
                 X_neg, y_neg, X_pos, y_pos, verbose=1,
                 use_vad_eval=False, frame_data_dict=None,
                 positive_frames=None, negative_frames=None,
                 threshold_good=0.5, threshold_bad=1.0, fitness_type='discrete',
                 rep_method='Arachne_v1', output_dir=None,
                 evaluate_fitness_fn=None, evaluate_fitness_vad_fn=None,
                 apply_weights_fn=None):
        """
        Run PSO optimization.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model to optimize
        particles : list
            Initial particles
        weights_to_repair : list
            List of weights to repair
        X_neg, y_neg : torch.Tensor
            Negative samples
        X_pos, y_pos : torch.Tensor
            Positive samples
        verbose : int
            Verbosity level
        use_vad_eval : bool
            Whether to use VAD evaluation
        frame_data_dict : dict
            Frame data for VAD evaluation
        positive_frames, negative_frames : list
            Frame lists for VAD evaluation
        threshold_good, threshold_bad : float
            Thresholds for VAD evaluation
        fitness_type : str
            Type of fitness function
        rep_method : str
            Repair method name
        output_dir : Path
            Directory to save results
        evaluate_fitness_fn : callable
            Function to evaluate fitness (standard)
        evaluate_fitness_vad_fn : callable
            Function to evaluate fitness (VAD)
        apply_weights_fn : callable
            Function to apply weights to model
        
        Returns
        -------
        best_model : nn.Module
            Best model found
        fitness_history : list
            History of best fitness values
        """
        global_best_position = None
        global_best_fitness = float('inf')
        global_best_model = None
        fitness_history = []
        particle_evaluations = []
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Evaluate original model
        print("\nEvaluating original model...", flush=True)
        frame_counts_for_logging = None
        if use_vad_eval:
            # For original model evaluation, use original L2 from JSON for classification consistency
            result = evaluate_fitness_vad_fn(
                model, frame_data_dict,
                positive_frames, negative_frames,
                threshold_good, threshold_bad, fitness_type, rep_method,
                use_original_l2_for_classification=True
            )
            # Handle both return formats: (fitness, frame_counts) or just fitness
            if isinstance(result, tuple) and len(result) == 2:
                original_fitness, frame_counts_for_logging = result
            else:
                original_fitness = result
        else:
            original_fitness = evaluate_fitness_fn(
                model, X_neg, y_neg, X_pos, y_pos, loss_fn
            )
        
        print(f"Original model fitness: {original_fitness:.6f}", flush=True)
        if frame_counts_for_logging is not None:
            print(f"  Frame counts for fitness calculation:", flush=True)
            print(f"    Positive (no collision): {frame_counts_for_logging['positive_no_collision']}", flush=True)
            print(f"    Middle (no collision): {frame_counts_for_logging['middle_no_collision']}", flush=True)
            print(f"    Negative (no collision): {frame_counts_for_logging['negative_no_collision']}", flush=True)
            print(f"    Collision: {frame_counts_for_logging['collision']}", flush=True)
            print(f"    Total evaluated: {frame_counts_for_logging['total_evaluated']}", flush=True)
        fitness_history.append(original_fitness)
        
        # Initialize global best
        global_best_fitness = original_fitness
        global_best_position = {}
        named_modules_for_best = dict(model.named_modules())
        for weight_info in weights_to_repair:
            layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
            key = (layer_name, i, j)
            if layer_name in named_modules_for_best:
                layer = named_modules_for_best[layer_name]
                global_best_position[key] = float(layer.weight[j, i].item())
        global_best_model = copy.deepcopy(model)
        
        # Pre-compute bounds for each layer (original method: symmetric extension)
        # Same as old PSO implementation: extend range by 50% on each side
        named_modules = dict(model.named_modules())
        weight_ranges = {}
        
        # Compute bounds for each unique layer
        for weight_info in weights_to_repair:
            layer_name = weight_info[0]
            if layer_name not in weight_ranges:
                if layer_name in named_modules:
                    layer = named_modules[layer_name]
                    layer_weights = layer.weight.detach().cpu().numpy()
                    weight_min = float(np.min(layer_weights))
                    weight_max = float(np.max(layer_weights))
                    weight_range = weight_max - weight_min
                    # Original method: symmetric extension by 50% on each side
                    bound_min = weight_min - weight_range * 0.5
                    bound_max = weight_max + weight_range * 0.5
                    weight_ranges[layer_name] = {
                        'min': bound_min,
                        'max': bound_max,
                        'velocity_max': weight_range * 0.5  # For PSO velocity clamping
                    }
                else:
                    # Fallback
                    weight_ranges[layer_name] = {
                        'min': -2.0,
                        'max': 2.0,
                        'velocity_max': 1.0
                    }
        
        # Early stopping variables
        best_fitness_so_far = global_best_fitness
        best_iteration_so_far = 0
        no_improvement_count = 0
        prev_iteration_fitness = None
        
        # PSO optimization loop
        for iteration in range(self.num_iterations):
            for p_idx, particle in enumerate(particles):
                # Apply particle's position to model
                modified_model = apply_weights_fn(
                    model, weights_to_repair, particle['position']
                )
                
                # Evaluate fitness
                if use_vad_eval:
                    fitness = evaluate_fitness_vad_fn(
                        modified_model, frame_data_dict,
                        positive_frames, negative_frames,
                        threshold_good, threshold_bad, fitness_type, rep_method
                    )
                else:
                    fitness = evaluate_fitness_fn(
                        modified_model, X_neg, y_neg, X_pos, y_pos, loss_fn
                    )
                
                # Update particle's best
                is_particle_best = (fitness < particle['best_fitness'])
                if is_particle_best:
                    particle['best_fitness'] = fitness
                    particle['best_position'] = copy.deepcopy(particle['position'])
                
                # Update global best
                is_global_best = (fitness < global_best_fitness)
                if is_global_best:
                    global_best_fitness = fitness
                    global_best_position = copy.deepcopy(particle['position'])
                    global_best_model = copy.deepcopy(modified_model)
                
                # Record evaluation
                position_serializable = {}
                for key, value in particle['position'].items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        position_serializable[str(key)] = float(value)
                    else:
                        position_serializable[str(key)] = float(value)
                
                particle_evaluations.append({
                    'iteration': iteration + 1,
                    'particle_idx': p_idx,
                    'fitness': float(fitness),
                    'position': position_serializable,
                    'is_global_best': is_global_best,
                    'is_particle_best': is_particle_best
                })
            
            # Record best fitness for this iteration
            fitness_history.append(global_best_fitness)
            
            # Early stopping check
            if self.early_stop_patience is not None and self.early_stop_patience > 0:
                if global_best_fitness < best_fitness_so_far:
                    best_fitness_so_far = global_best_fitness
                    best_iteration_so_far = iteration + 1
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= self.early_stop_patience:
                    print(f"\nEarly stopping triggered: No improvement for {self.early_stop_patience} iterations", flush=True)
                    print(f"  Best fitness: {best_fitness_so_far:.6f} (achieved at iteration {best_iteration_so_far})", flush=True)
                    print(f"  Stopped at iteration {iteration + 1}/{self.num_iterations}", flush=True)
                    break
            
            # Print iteration info
            if prev_iteration_fitness is not None:
                delta = prev_iteration_fitness - global_best_fitness
            elif len(fitness_history) > 1:
                delta = fitness_history[-2] - fitness_history[-1] if len(fitness_history) >= 2 else 0.0
            else:
                delta = 0.0
            
            prev_iteration_fitness = global_best_fitness
            
            if len(fitness_history) > 0:
                early_stop_msg = f", no_improvement={no_improvement_count}/{self.early_stop_patience}" if (self.early_stop_patience is not None and verbose) else ""
                # Debug: show if fitness changed in this iteration
                fitness_changed = (delta != 0.0) if prev_iteration_fitness is not None else False
                change_indicator = " ✓" if fitness_changed else ""
                print(f"Iter {iteration+1:3d}/{self.num_iterations}: fitness={global_best_fitness:7.2f}, Δ={delta:+6.2f}{change_indicator}{early_stop_msg}", flush=True)
            
            # Update velocities and positions
            for particle in particles:
                for weight_info in weights_to_repair:
                    layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                    key = (layer_name, i, j)
                    
                    if key not in particle['position']:
                        continue
                    
                    layer_range = weight_ranges.get(layer_name, {'min': -2.0, 'max': 2.0, 'velocity_max': 1.0})
                    
                    # PSO velocity update
                    r1, r2 = np.random.rand(), np.random.rand()
                    cognitive = 2.0 * r1 * (particle['best_position'][key] - particle['position'][key])
                    global_best_val = global_best_position.get(key, particle['position'][key])
                    social = 2.0 * r2 * (global_best_val - particle['position'][key])
                    
                    # Apply inertia weight
                    particle['velocity'][key] = (self.velocity_phi * particle['velocity'][key] + 
                                                cognitive + social)
                    
                    # Clamp velocity
                    velocity_max = layer_range['velocity_max']
                    particle['velocity'][key] = np.clip(particle['velocity'][key], -velocity_max, velocity_max)
                    
                    # Position update
                    old_position = particle['position'][key]
                    particle['position'][key] += particle['velocity'][key]
                    new_position_before_clamp = particle['position'][key]
                    
                    # Clamp position
                    new_position = np.clip(
                        particle['position'][key], 
                        layer_range['min'], 
                        layer_range['max']
                    )
                    
                    # Adjust velocity if position was clamped
                    if abs(new_position - new_position_before_clamp) > 1e-10:
                        if abs(new_position - layer_range['min']) < 1e-10:
                            particle['velocity'][key] = -particle['velocity'][key] * 0.5
                        elif abs(new_position - layer_range['max']) < 1e-10:
                            particle['velocity'][key] = -particle['velocity'][key] * 0.5
                    
                    particle['position'][key] = new_position
        
        print(f"\nPSO optimization complete!", flush=True)
        print(f"  Original fitness: {fitness_history[0]:.6f}", flush=True)
        print(f"  Final fitness:    {fitness_history[-1]:.6f}", flush=True)
        print(f"  Improvement:      {fitness_history[0] - fitness_history[-1]:.6f}", flush=True)
        if fitness_history[0] > 0:
            print(f"  Reduction:        {(1 - fitness_history[-1]/fitness_history[0])*100:.1f}%", flush=True)
        
        # Save particle evaluations
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            particles_file = output_dir / 'particle_evaluations.json'
            with open(particles_file, 'w') as f:
                json.dump({
                    'total_evaluations': len(particle_evaluations),
                    'num_iterations': len(fitness_history) - 1,
                    'num_particles': len(particles),
                    'fitness_history': fitness_history,
                    'particle_evaluations': particle_evaluations
                }, f, indent=2)
            print(f"  Particle evaluations saved to: {particles_file}")
            print(f"    Total evaluations: {len(particle_evaluations)}")
        
        # Ensure we return a valid model (should never be None, but check just in case)
        if global_best_model is None:
            print("  [WARNING] global_best_model is None! Using original model as fallback.", flush=True)
            global_best_model = copy.deepcopy(model)
        
        return global_best_model, fitness_history

