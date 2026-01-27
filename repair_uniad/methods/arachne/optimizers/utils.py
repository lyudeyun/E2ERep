"""
Utility functions shared by optimization algorithms.
"""

import torch
import torch.nn as nn
import numpy as np


def compute_weight_ranges(model, weights_to_repair):
    """
    Compute weight ranges for each layer.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    weights_to_repair : list
        List of weights to repair
    
    Returns
    -------
    weight_ranges : dict
        Dictionary mapping layer_name to {'min', 'max', 'velocity_max'}
    """
    weight_ranges = {}
    named_modules = dict(model.named_modules())
    
    for weight_info in weights_to_repair:
        layer_name = weight_info[0]
        if layer_name not in weight_ranges:
            if layer_name in named_modules:
                layer = named_modules[layer_name]
                layer_weights = layer.weight.detach().cpu().numpy()
                weight_min = float(np.min(layer_weights))
                weight_max = float(np.max(layer_weights))
                weight_range = weight_max - weight_min
                weight_ranges[layer_name] = {
                    'min': weight_min - weight_range * 0.5,
                    'max': weight_max + weight_range * 0.5,
                    'velocity_max': weight_range * 0.5  # For PSO
                }
            else:
                weight_ranges[layer_name] = {
                    'min': -2.0,
                    'max': 2.0,
                    'velocity_max': 1.0
                }
    
    return weight_ranges


def initialize_weight_position(model, weights_to_repair, weight_ranges):
    """
    Initialize weight positions near original weights.
    
    Parameters
    ----------
    model : nn.Module
        PyTorch model
    weights_to_repair : list
        List of weights to repair
    weight_ranges : dict
        Weight ranges for each layer
    
    Returns
    -------
    position : dict
        Dictionary mapping (layer_name, i, j) to weight value
    """
    position = {}
    named_modules = dict(model.named_modules())
    
    for weight_info in weights_to_repair:
        layer_name, i, j = weight_info[0], weight_info[1], weight_info[2]
        key = (layer_name, i, j)
        
        # Get original weight
        if layer_name in named_modules:
            layer = named_modules[layer_name]
            original_weight = float(layer.weight[j, i].item())
        else:
            layer_info = weight_ranges.get(layer_name, {'min': -1.0, 'max': 1.0})
            original_weight = (layer_info['min'] + layer_info['max']) / 2.0
        
        # Initialize near original with small perturbation
        layer_info = weight_ranges.get(layer_name, {'min': -1.0, 'max': 1.0, 'range': 2.0})
        weight_min = layer_info['min']
        weight_max = layer_info['max']
        weight_range = layer_info.get('range', weight_max - weight_min)
        
        perturbation_range = weight_range * 0.1
        initial_weight = original_weight + np.random.uniform(-perturbation_range, perturbation_range)
        initial_weight = np.clip(initial_weight, weight_min, weight_max)
        
        position[key] = initial_weight
    
    return position

