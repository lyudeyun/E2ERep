#!/usr/bin/env python3
"""
Evaluation results analysis script

Usage:
1. Modify the JSON_FILES list in the script to add filenames to analyze
2. Run: python3 analyze_evaluation_results.py

Output:
- Results will be saved to CSV file: evaluation_results.csv
- Also displayed in terminal (optional)

Output format:
- First line: filename + success/failure status for all routes + success rate
- Second line: Route Complete + completion rate for all routes + average
- Third line: Driving Score + driving score for all routes + average
"""

import json
import os
import sys
import csv
from datetime import datetime

# ============================================================================
# Configuration: Add JSON filenames to analyze here
# ============================================================================
JSON_FILES = [
    "cl_vad_tiny_Arachne_v1_mid_results_CONT2_max.json",
    "cl_vad_tiny_Arachne_v1_mid_results_CONT2_median.json",
]

# ============================================================================
# Main functions
# ============================================================================

def is_success(status):
    """Check if route is successful"""
    return status == "Completed"

def analyze_json_file(filepath):
    """Analyze a single JSON file"""
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}", file=sys.stderr)
        return None
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to read file {filepath}: {e}", file=sys.stderr)
        return None
    
    checkpoint = data.get('_checkpoint', {})
    records = checkpoint.get('records', [])
    
    if len(records) == 0:
        print(f"WARNING: File {filepath} has no records data", file=sys.stderr)
        return None
    
    # Extract data
    success_failure = []
    route_complete = []
    driving_score = []
    
    for record in records:
        status = record.get('status', 'Unknown')
        scores = record.get('scores', {})
        
        # Success/failure status
        success_failure.append(is_success(status))
        
        # Route Complete (score_route)
        route_complete.append(scores.get('score_route', 0.0))
        
        # Driving Score (score_composed)
        driving_score.append(scores.get('score_composed', 0.0))
    
    # Calculate statistics
    success_count = sum(success_failure)
    success_rate = (success_count / len(success_failure) * 100) if success_failure else 0.0
    avg_route_complete = sum(route_complete) / len(route_complete) if route_complete else 0.0
    avg_driving_score = sum(driving_score) / len(driving_score) if driving_score else 0.0
    
    return {
        'filename': os.path.basename(filepath),
        'success_failure': success_failure,
        'route_complete': route_complete,
        'driving_score': driving_score,
        'success_rate': success_rate,
        'avg_route_complete': avg_route_complete,
        'avg_driving_score': avg_driving_score
    }

def format_output(result):
    """Format output"""
    # Extract filename, remove cl_ prefix and .json suffix, simplify format
    filename = result['filename'].replace('cl_', '').replace('.json', '')
    # Optional: further simplify filename format if needed
    # filename = filename.replace('_results', '')
    
    # First line: filename + success/failure status + success rate
    success_failure_str = '\t'.join(['PASS' if s else 'FAIL' for s in result['success_failure']])
    line1 = f"{filename}\tSuccess or failure\t{success_failure_str}\t{result['success_rate']:.0f}%"
    
    # Format numbers: display as integer if whole number, otherwise 2 decimal places
    def format_number(num):
        if num == int(num):
            return str(int(num))
        else:
            return f"{num:.2f}"
    
    # Second line: Route Complete + scores + average
    route_complete_str = '\t'.join([format_number(score) for score in result['route_complete']])
    line2 = f"\tRoute Complete\t{route_complete_str}\t{format_number(result['avg_route_complete'])}"
    
    # Third line: Driving Score + scores + average
    driving_score_str = '\t'.join([format_number(score) for score in result['driving_score']])
    line3 = f"\tDriving Score\t{driving_score_str}\t{format_number(result['avg_driving_score'])}"
    
    return f"{line1}\n{line2}\n{line3}"

def save_to_csv(results, output_file="evaluation_results.csv"):
    """Save results to CSV file"""
    if not results:
        return
    
    # Find the maximum number of routes across all results
    max_routes = max(len(result['success_failure']) for result in results)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t')
        
        # Write header dynamically based on max routes
        header = ['Task Name', 'Metric']
        header.extend([f'Route_{i+1}' for i in range(max_routes)])
        header.append('Average/Total')
        writer.writerow(header)
        
        # Write data
        for result in results:
            filename = result['filename'].replace('cl_', '').replace('.json', '')
            num_routes = len(result['success_failure'])
            
            # First row: success/failure status
            success_failure_row = [filename, 'Success or failure']
            success_failure_row.extend(['PASS' if s else 'FAIL' for s in result['success_failure']])
            # Pad with empty strings if fewer routes than max
            success_failure_row.extend([''] * (max_routes - num_routes))
            success_failure_row.append(f"{result['success_rate']:.0f}%")
            writer.writerow(success_failure_row)
            
            # Second row: Route Complete
            route_complete_row = ['', 'Route Complete']
            route_complete_row.extend([f"{score:.2f}" for score in result['route_complete']])
            # Pad with empty strings if fewer routes than max
            route_complete_row.extend([''] * (max_routes - num_routes))
            route_complete_row.append(f"{result['avg_route_complete']:.2f}")
            writer.writerow(route_complete_row)
            
            # Third row: Driving Score
            driving_score_row = ['', 'Driving Score']
            driving_score_row.extend([f"{score:.2f}" for score in result['driving_score']])
            # Pad with empty strings if fewer routes than max
            driving_score_row.extend([''] * (max_routes - num_routes))
            driving_score_row.append(f"{result['avg_driving_score']:.2f}")
            writer.writerow(driving_score_row)
            
            # Add empty row to separate different tasks
            writer.writerow([])

def main():
    """Main function"""
    if len(JSON_FILES) == 0:
        print("WARNING: Please configure JSON_FILES list in the script", file=sys.stderr)
        sys.exit(1)
    
    results = []
    for json_file in JSON_FILES:
        result = analyze_json_file(json_file)
        if result:
            results.append(result)
    
    if len(results) == 0:
        print("ERROR: No files were successfully analyzed", file=sys.stderr)
        sys.exit(1)
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evaluation_results_{timestamp}.csv"
    
    # Save to CSV
    save_to_csv(results, output_file)
    print(f"Results saved to: {output_file}")
    print()
    
    # Also display in terminal (optional)
    print("=" * 100)
    print("Evaluation Results Summary")
    print("=" * 100)
    print()
    
    for result in results:
        print(format_output(result))
        print()
    
    print(f"\nCSV file saved: {output_file}")

if __name__ == "__main__":
    main()
