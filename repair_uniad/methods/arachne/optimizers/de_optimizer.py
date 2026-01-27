"""
Differential Evolution (DE) optimizer for neural network repair.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import copy
# Note: compute_weight_ranges and initialize_weight_position are no longer used
# (replaced with Arachne v2 initialization method)


class DEOptimizer:
    """
    Differential Evolution optimizer.
    
    DE uses mutation, crossover, and selection operations to evolve
    a population of candidate solutions.
    """
    
    def __init__(self, num_iterations=100, num_particles=100,
                 de_crossover_rate=0.9, de_mutation_factor=0.8,
                 de_strategy='rand/1/bin', early_stop_patience=None,
                 device=None, eval_batch_size=None):
        """
        Initialize DE optimizer.
        
        Parameters
        ----------
        num_iterations : int
            Maximum number of iterations
        num_particles : int
            Population size (number of individuals)
        de_crossover_rate : float
            Crossover rate (CR) in [0, 1], typically 0.9
        de_mutation_factor : float
            Mutation factor (F) in [0, 2], typically 0.5-1.0
        de_strategy : str
            DE strategy: 'rand/1/bin', 'best/1/bin', 'rand/2/bin', etc.
        early_stop_patience : int or None
            Early stopping patience
        device : torch.device
            Device to run optimization on
        eval_batch_size : int or None
            Batch size for fitness evaluation
        """
        self.num_iterations = num_iterations
        self.num_particles = num_particles
        self.de_crossover_rate = de_crossover_rate
        self.de_mutation_factor = de_mutation_factor
        self.de_strategy = de_strategy
        self.early_stop_patience = early_stop_patience
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.eval_batch_size = eval_batch_size
    
    def initialize_population(self, model, weights_to_repair):
        """
        Initialize DE population (individuals) using original weight values with small perturbations.
        
        For each weight to repair, initialize near its original value with small random perturbation.
        This ensures initial population is close to the original model, providing better starting point.
        
        Parameters
        ----------
        model : nn.Module
            PyTorch model
        weights_to_repair : list
            List of weights to repair [layer_name, i, j, GL, FI]
        
        Returns
        -------
        individuals : list
            List of individual dictionaries
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
                    weight_mean = float(np.mean(layer_weights))
                    weight_std = float(np.std(layer_weights))
                    # Store extended bounds (larger than VAD for better exploration)
                    layer_ranges[layer_name] = {
                        'min': weight_min - weight_range * 0.5,  # Extended bound
                        'max': weight_max + weight_range * 0.5,  # Extended bound
                        'range': weight_range,  # Original range
                        'original_min': weight_min,
                        'original_max': weight_max,
                        'original_range': weight_range
                    }
                    # Print weight statistics for debugging
                    print(f"\nWeight statistics for layer '{layer_name}':", flush=True)
                    print(f"  Shape: {layer.weight.shape}", flush=True)
                    print(f"  Min: {weight_min:.6f}", flush=True)
                    print(f"  Max: {weight_max:.6f}", flush=True)
                    print(f"  Range: {weight_range:.6f} (Max - Min)", flush=True)
                    print(f"  Mean: {weight_mean:.6f}", flush=True)
                    print(f"  Std: {weight_std:.6f}", flush=True)
                    print(f"  Search bounds: [{weight_min - weight_range * 0.5:.6f}, {weight_max + weight_range * 0.5:.6f}] (extended by ±50%)", flush=True)
                    print(f"  Initialization perturbation: ±{weight_range * 0.1:.6f} (10% of range, same as VAD/PSO)", flush=True)
                else:
                    # Fallback
                    layer_ranges[layer_name] = {'min': -1.0, 'max': 1.0, 'range': 2.0, 'original_min': -1.0, 'original_max': 1.0, 'original_range': 2.0}
                    print(f"\nWarning: Layer '{layer_name}' not found, using fallback ranges", flush=True)
        
        # Initialize population
        individuals = []
        for _ in range(self.num_particles):
            # Initialize near original weights with small perturbations
            position = {}
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
                
                # Use same perturbation as VAD/PSO: 10% of weight range
                perturbation_range = weight_range * 0.1  # 10% of weight range (same as VAD/PSO)
                initial_weight = original_weight + np.random.uniform(-perturbation_range, perturbation_range)
                initial_weight = np.clip(initial_weight, weight_min, weight_max)
                
                
                position[key] = float(initial_weight)
            
            individual = {
                'position': position,
                'fitness': float('inf'),
                'best_fitness': float('inf')
            }
            individuals.append(individual)
        
        return individuals
    
    def optimize(self, model, individuals, weights_to_repair,
                 X_neg, y_neg, X_pos, y_pos, verbose=1,
                 use_cached_eval=False, frame_data_dict=None,
                 positive_frames=None, negative_frames=None,
                 threshold_good=0.5, threshold_bad=1.0, fitness_type='discrete',
                 rep_method='Arachne_v1', output_dir=None,
                 evaluate_fitness_fn=None, evaluate_fitness_openloop_fn=None,
                 apply_weights_fn=None, time_horizon=3):
        """
        Run DE optimization.
        
        Parameters are same as PSOOptimizer.optimize()
        
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
        individual_evaluations = []
        
        loss_fn = nn.BCEWithLogitsLoss()
        
        # Evaluate original model
        print("\nEvaluating original model...", flush=True)
        frame_counts_for_logging = None
        if use_cached_eval:
            # For original model evaluation, use original L2 from JSON for classification consistency
            result = evaluate_fitness_openloop_fn(
                model, frame_data_dict,
                positive_frames, negative_frames,
                threshold_good, threshold_bad, fitness_type, rep_method,
                use_original_l2_for_classification=True, time_horizon=time_horizon
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
        
        
        # Get perturbation_range from initialize_population to check if it's zero
        # This is used to determine if we should use JSON L2 for initial population
        # Note: With 10% perturbation (same as VAD/PSO), weights are NOT identical to original,
        # so we should always use computed L2 (not JSON L2) for individual evaluation
        perturbation_range = None
        # Check the first individual to get perturbation_range (it's set in initialize_population)
        if len(individuals) > 0:
            # perturbation_range is now 10% of weight range (same as VAD/PSO), not 0.0
            # We need to compute it from the first weight's range
            # For simplicity, we'll use a small threshold to detect near-zero perturbation
            # Since we're using 10% perturbation now, this will be False
            perturbation_range = 0.1  # 10% of weight range (approximate, actual value varies per layer)
        
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
        
        # Pre-compute bounds for each weight to repair (original method: symmetric extension)
        # For each weight, use its layer's entire weight matrix min/max
        # Then extend symmetrically by 50% on each side (same as old PSO implementation)
        named_modules = dict(model.named_modules())
        bounds = []  # List of (min, max) for each weight to repair
        
        for weight_info in weights_to_repair:
            layer_name = weight_info[0]
            if layer_name in named_modules:
                layer = named_modules[layer_name]
                layer_weights = layer.weight.detach().cpu().numpy()
                weight_min = float(np.min(layer_weights))
                weight_max = float(np.max(layer_weights))
                weight_range = weight_max - weight_min
                # Original method: symmetric extension by 50% on each side
                bound_min = weight_min - weight_range * 0.5
                bound_max = weight_max + weight_range * 0.5
                bounds.append((bound_min, bound_max))
            else:
                # Fallback
                bounds.append((-2.0, 2.0))
        
        # Evaluate initial population
        print("Evaluating initial DE population...", flush=True)
        import time
        total_start_time = time.time()
        initial_fitnesses = []
        for idx, individual in enumerate(individuals):
            
            modified_model = apply_weights_fn(
                model, weights_to_repair, individual['position']
            )
            
            
            if use_cached_eval:
                # For initial population evaluation: use computed L2 to reflect actual model changes
                # Even though initial population has only 10% perturbation, we need to use computed L2
                # so that fitness reflects the weight changes. Using JSON L2 would make all individuals
                # have identical fitness (6749.27) regardless of weight changes, which is incorrect.
                # Note: Computed L2 may give different absolute values than JSON L2, but the relative
                # differences between individuals will be correct.
                use_json_l2_for_individual = False  # Use computed L2 for initial population to reflect weight changes
                
                result = evaluate_fitness_openloop_fn(
                    modified_model, frame_data_dict,
                    positive_frames, negative_frames,
                    threshold_good, threshold_bad, fitness_type, rep_method,
                    use_original_l2_for_classification=use_json_l2_for_individual,  # Use JSON L2 for initial population
                    time_horizon=time_horizon
                )
                # Handle both return formats: (fitness, frame_counts) or just fitness
                if isinstance(result, tuple) and len(result) == 2:
                    fitness, frame_counts_for_individual = result
                else:
                    fitness = result
                    frame_counts_for_individual = None
                
            else:
                fitness = evaluate_fitness_fn(
                    modified_model, X_neg, y_neg, X_pos, y_pos, loss_fn
                )
            
            
            individual['fitness'] = fitness
            individual['best_fitness'] = fitness
            initial_fitnesses.append(fitness)
            
            # Update global best
            
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = copy.deepcopy(individual['position'])
                global_best_model = copy.deepcopy(modified_model)
            
            # Print progress every 10 individuals
            if (idx + 1) % 10 == 0 or (idx + 1) == len(individuals):
                elapsed = time.time() - total_start_time
                avg_time_per_individual = elapsed / (idx + 1)
                remaining = (len(individuals) - idx - 1) * avg_time_per_individual
                print(f"  Progress: {idx+1}/{len(individuals)} individuals evaluated "
                      f"(avg {avg_time_per_individual:.2f}s/individual, "
                      f"~{remaining:.0f}s remaining)", flush=True)
        
        fitness_history.append(global_best_fitness)
        print(f"Initial population best fitness: {global_best_fitness:.6f}", flush=True)
        # Early stopping variables
        best_fitness_so_far = global_best_fitness
        best_iteration_so_far = 0
        no_improvement_count = 0
        
        # DE optimization loop
        for iteration in range(self.num_iterations):
            # Arachne v2: Randomly select mutation factor MU once per iteration
            # All individuals in this iteration use the same MU value
            if isinstance(self.de_mutation_factor, (tuple, list)) and len(self.de_mutation_factor) == 2:
                MU = np.random.uniform(self.de_mutation_factor[0], self.de_mutation_factor[1])
            else:
                MU = self.de_mutation_factor
            
            new_population = []
            improvements = 0
            
            for idx, individual in enumerate(individuals):
                # Select 3 different random individuals
                candidates = [i for i in range(len(individuals)) if i != idx]
                r1_idx, r2_idx, r3_idx = np.random.choice(candidates, 3, replace=False)
                r1 = individuals[r1_idx]
                r2 = individuals[r2_idx]
                r3 = individuals[r3_idx]
                
                # Generate mutant vector based on strategy (Arachne v2 logic)
                # Use the MU value selected at the start of this iteration
                
                if self.de_strategy == 'rand/1/bin':
                    # v = r1 + MU * (r2 - r3)  (Arachne v2 style)
                    mutant = {}
                    for weight_info in weights_to_repair:
                        layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                        key = (layer_name, i, j)
                        if key in r1['position'] and key in r2['position'] and key in r3['position']:
                            mutant[key] = (r1['position'][key] + 
                                         MU * 
                                         (r2['position'][key] - r3['position'][key]))
                        else:
                            mutant[key] = individual['position'].get(key, 0.0)
                
                elif self.de_strategy == 'best/1/bin':
                    # v = best + MU * (r1 - r2)  (Arachne v2 style)
                    mutant = {}
                    for weight_info in weights_to_repair:
                        layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                        key = (layer_name, i, j)
                        best_val = global_best_position.get(key, individual['position'].get(key, 0.0))
                        if key in r1['position'] and key in r2['position']:
                            mutant[key] = (best_val + 
                                         MU * 
                                         (r1['position'][key] - r2['position'][key]))
                        else:
                            mutant[key] = individual['position'].get(key, 0.0)
                
                else:
                    # Default to rand/1/bin (Arachne v2 style)
                    mutant = {}
                    for weight_info in weights_to_repair:
                        layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                        key = (layer_name, i, j)
                        if key in r1['position'] and key in r2['position'] and key in r3['position']:
                            mutant[key] = (r1['position'][key] + 
                                         MU * 
                                         (r2['position'][key] - r3['position'][key]))
                        else:
                            mutant[key] = individual['position'].get(key, 0.0)
                
                # Apply bounds to mutant (Arachne v2 style: per-weight bounds)
                for idx, weight_info in enumerate(weights_to_repair):
                    layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                    key = (layer_name, i, j)
                    if key in mutant:
                        bound_min, bound_max = bounds[idx]
                        mutant[key] = np.clip(mutant[key], bound_min, bound_max)
                
                # Binomial crossover to create trial vector (Arachne v2 style)
                # v2 uses: if i == index or random() < CR, use mutant[i], else use individual[i]
                trial = {}
                j_rand = np.random.randint(len(weights_to_repair))
                
                for dim_idx, weight_info in enumerate(weights_to_repair):
                    layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
                    key = (layer_name, i, j)
                    
                    # Arachne v2 crossover logic: ensure at least one dimension from mutant
                    if dim_idx == j_rand or np.random.rand() < self.de_crossover_rate:
                        trial[key] = mutant.get(key, individual['position'].get(key, 0.0))
                    else:
                        trial[key] = individual['position'].get(key, 0.0)
                
                # Evaluate trial vector
                trial_model = apply_weights_fn(
                    model, weights_to_repair, trial
                )
                
                if use_cached_eval:
                    # For optimization loop: use computed L2 to reflect actual model changes
                    # This is different from initial population which uses JSON L2 for consistency
                    trial_fitness = evaluate_fitness_openloop_fn(
                        trial_model, frame_data_dict,
                        positive_frames, negative_frames,
                        threshold_good, threshold_bad, fitness_type, rep_method,
                        use_original_l2_for_classification=False,  # Use computed L2 in optimization loop
                        time_horizon=time_horizon
                    )
                else:
                    trial_fitness = evaluate_fitness_fn(
                        trial_model, X_neg, y_neg, X_pos, y_pos, loss_fn
                    )
                
                # Selection: greedy selection
                # improvements is relative to current individual's fitness, not original model
                is_improvement = trial_fitness < individual['fitness']
                if is_improvement:
                    new_individual = {
                        'position': copy.deepcopy(trial),
                        'fitness': trial_fitness,
                        'best_fitness': trial_fitness
                    }
                    improvements += 1
                    
                    # Update global best
                    if trial_fitness < global_best_fitness:
                        global_best_fitness = trial_fitness
                        global_best_position = copy.deepcopy(trial)
                        global_best_model = copy.deepcopy(trial_model)
                else:
                    new_individual = copy.deepcopy(individual)
                
                new_population.append(new_individual)
                
                # Record evaluation
                position_serializable = {}
                for key, value in new_individual['position'].items():
                    if isinstance(value, (np.ndarray, np.generic)):
                        position_serializable[str(key)] = float(value)
                    else:
                        position_serializable[str(key)] = float(value)
                
                individual_evaluations.append({
                    'iteration': iteration + 1,
                    'individual_idx': idx,
                    'fitness': float(new_individual['fitness']),
                    'position': position_serializable,
                    'is_improvement': is_improvement
                })
            
            # Update population
            individuals = new_population
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
            prev_fitness = fitness_history[-2] if len(fitness_history) >= 2 else original_fitness
            delta = prev_fitness - global_best_fitness
            improvement = original_fitness - global_best_fitness
            
            
            early_stop_msg = f", no_improvement={no_improvement_count}/{self.early_stop_patience}" if (self.early_stop_patience is not None and verbose) else ""
            print(f"Iter {iteration+1:3d}/{self.num_iterations}: fitness={global_best_fitness:7.2f}, "
                  f"Δ={delta:+6.2f}, improvements={improvements}/{len(individuals)} (vs current individuals, not original){early_stop_msg}", flush=True)
        
        print(f"\nDE optimization complete!", flush=True)
        print(f"  Original fitness: {fitness_history[0]:.6f}", flush=True)
        print(f"  Final fitness:    {fitness_history[-1]:.6f}", flush=True)
        print(f"  Improvement:      {fitness_history[0] - fitness_history[-1]:.6f}", flush=True)
        if fitness_history[0] > 0:
            print(f"  Reduction:        {(1 - fitness_history[-1]/fitness_history[0])*100:.1f}%", flush=True)
        
        # Save individual evaluations
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            individuals_file = output_dir / 'de_evaluations.json'
            with open(individuals_file, 'w') as f:
                json.dump({
                    'total_evaluations': len(individual_evaluations),
                    'num_iterations': len(fitness_history) - 1,
                    'num_individuals': len(individuals),
                    'de_strategy': self.de_strategy,
                    'de_crossover_rate': self.de_crossover_rate,
                    'de_mutation_factor': self.de_mutation_factor,
                    'individual_evaluations': individual_evaluations
                }, f, indent=2)
            print(f"  DE evaluations saved to: {individuals_file}")
        
        return global_best_model, fitness_history

