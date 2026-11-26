#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot L2 error 3s distribution from baseline JSON file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_l2_errors(json_path):
    """Load plan_L2_3s values from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    l2_errors = []
    for entry in data:
        # Only include entries with valid future trajectory
        if entry.get('fut_valid_flag', False):
            l2_3s = entry.get('plan_L2_3s')
            if l2_3s is not None and l2_3s >= 0:
                l2_errors.append(l2_3s)
    
    return np.array(l2_errors)

def plot_distribution(l2_errors, output_path=None):
    """Plot L2 error distribution with statistical markers."""
    # Calculate statistics
    mean = np.mean(l2_errors)
    std = np.std(l2_errors)
    median = np.median(l2_errors)
    
    print(f"Statistics (valid frames only, fut_valid_flag=True):")
    print(f"  Count: {len(l2_errors)}")
    print(f"  Mean: {mean:.6f}")
    print(f"  Std: {std:.6f}")
    print(f"  Median: {np.median(l2_errors):.6f}")
    print(f"  Mean ± 0.5*std: [{mean - 0.5*std:.6f}, {mean + 0.5*std:.6f}]")
    print(f"  Mean ± 1*std: [{mean - 1*std:.6f}, {mean + 1*std:.6f}]")
    print(f"  Mean ± 2*std: [{mean - 2*std:.6f}, {mean + 2*std:.6f}]")
    print(f"\nSmall error counts:")
    print(f"  < 0.01: {np.sum(l2_errors < 0.01)} ({100*np.sum(l2_errors < 0.01)/len(l2_errors):.1f}%)")
    print(f"  < 0.1: {np.sum(l2_errors < 0.1)} ({100*np.sum(l2_errors < 0.1)/len(l2_errors):.1f}%)")
    print(f"  < 0.5: {np.sum(l2_errors < 0.5)} ({100*np.sum(l2_errors < 0.5)/len(l2_errors):.1f}%)")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot histogram (using count instead of density for more intuitive interpretation)
    n, bins, patches = ax.hist(l2_errors, bins=100, density=False, alpha=0.7, 
                               color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Note: KDE removed when using count instead of density, as KDE is a density estimate
    
    # Draw vertical lines for statistics
    colors = {
        'mean': 'green',
        'median': 'blue',
        'mean±0.5std': 'orange',
        'mean±1std': 'purple',
        'mean±2std': 'red'
    }
    
    # Median
    ax.axvline(median, color=colors['median'], linestyle='-', linewidth=2, 
               label=f'Median = {median:.6f}')
    
    # Mean
    ax.axvline(mean, color=colors['mean'], linestyle='--', linewidth=2, 
               label=f'Mean = {mean:.6f}')
    
    # Mean ± 0.5*std
    ax.axvline(mean - 0.5*std, color=colors['mean±0.5std'], linestyle=':', linewidth=1.5,
               label=f'Mean - 0.5*std = {mean - 0.5*std:.6f}')
    ax.axvline(mean + 0.5*std, color=colors['mean±0.5std'], linestyle=':', linewidth=1.5,
               label=f'Mean + 0.5*std = {mean + 0.5*std:.6f}')
    
    # Mean ± 1*std
    ax.axvline(mean - 1*std, color=colors['mean±1std'], linestyle='--', linewidth=1.5,
               label=f'Mean - 1*std = {mean - 1*std:.6f}')
    ax.axvline(mean + 1*std, color=colors['mean±1std'], linestyle='--', linewidth=1.5,
               label=f'Mean + 1*std = {mean + 1*std:.6f}')
    
    # Mean ± 2*std
    ax.axvline(mean - 2*std, color=colors['mean±2std'], linestyle='-.', linewidth=1.5,
               label=f'Mean - 2*std = {mean - 2*std:.6f}')
    ax.axvline(mean + 2*std, color=colors['mean±2std'], linestyle='-.', linewidth=1.5,
               label=f'Mean + 2*std = {mean + 2*std:.6f}')
    
    # Labels and title
    ax.set_xlabel('L2 Error (3s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count (Number of Frames)', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of L2 Error (3s) with Statistical Markers', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Add text box with statistics
    stats_text = f'Count: {len(l2_errors)} (valid frames only)\n'
    stats_text += f'Mean: {mean:.6f}\n'
    stats_text += f'Std: {std:.6f}\n'
    stats_text += f'Min: {l2_errors.min():.6f}\n'
    stats_text += f'Max: {l2_errors.max():.6f}\n'
    stats_text += f'Median: {np.median(l2_errors):.6f}\n'
    stats_text += f'\nSmall errors:\n'
    stats_text += f'< 0.01: {np.sum(l2_errors < 0.01)} ({100*np.sum(l2_errors < 0.01)/len(l2_errors):.1f}%)\n'
    stats_text += f'< 0.1: {np.sum(l2_errors < 0.1)} ({100*np.sum(l2_errors < 0.1)/len(l2_errors):.1f}%)\n'
    stats_text += f'< 0.5: {np.sum(l2_errors < 0.5)} ({100*np.sum(l2_errors < 0.5)/len(l2_errors):.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure
    if output_path is None:
        output_path = Path(json_path).parent / 'l2_error_3s_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()

if __name__ == '__main__':
    import sys
    
    json_path = Path('/home/deyun/git/E2ERep/baseline/vad_base_baseline_b2d_infos_val_partA_25clips.json')
    
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)
    
    print(f"Loading L2 errors from: {json_path}")
    l2_errors = load_l2_errors(json_path)
    
    if len(l2_errors) == 0:
        print("Error: No valid L2 error data found in JSON file")
        sys.exit(1)
    
    output_path = json_path.parent / 'l2_error_3s_distribution.png'
    plot_distribution(l2_errors, output_path)

