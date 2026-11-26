#!/usr/bin/env python3
"""
View CARLA evaluation results: Driving Score and Route Complete
Usage: python view_eval_results.py [eval.json]
"""

import json
import sys
import os
from pathlib import Path

def view_eval_results(json_file):
    """View evaluation results"""
    if not os.path.exists(json_file):
        print(f"ERROR: File not found: {json_file}")
        return
    
    with open(json_file) as f:
        data = json.load(f)
    
    # Display global statistics
    if '_checkpoint' in data and 'global_record' in data['_checkpoint']:
        global_record = data['_checkpoint']['global_record']
        if 'scores_mean' in global_record:
            print("=" * 60)
            print("Global Statistics")
            print("=" * 60)
            print(f"Driving Score:      {global_record['scores_mean']['score_composed']:.3f}")
            print(f"Route Complete:    {global_record['scores_mean']['score_route']:.3f}")
            print(f"Infraction Penalty: {global_record['scores_mean']['score_penalty']:.3f}")
            print()
    
    # Display detailed results for each route
    if '_checkpoint' in data and 'records' in data['_checkpoint']:
        records = data['_checkpoint']['records']
        print("=" * 60)
        print(f"Individual Route Results (Total: {len(records)} routes)")
        print("=" * 60)
        
        for i, record in enumerate(records):
            print(f"\nRoute {i+1}: {record['route_id']}")
            print(f"  Status:           {record['status']}")
            print(f"  Driving Score:    {record['scores']['score_composed']:.3f}")
            print(f"  Route Complete:   {record['scores']['score_route']:.3f}")
            print(f"  Infraction Penalty: {record['scores']['score_penalty']:.3f}")
            if 'save_name' in record:
                print(f"  Save Name:        {record['save_name']}")

def find_eval_json(eval_folder):
    """Find eval.json in the evaluation folder"""
    # Check eval_v1 directory
    eval_v1 = Path(__file__).parent / "eval_v1"
    if eval_v1.exists():
        eval_json = eval_v1 / "eval.json"
        if eval_json.exists():
            return str(eval_json)
    
    # Check current directory
    eval_json = Path(__file__).parent / "eval.json"
    if eval_json.exists():
        return str(eval_json)
    
    return None

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default: find eval.json
        json_file = find_eval_json("eval_v1")
        if not json_file:
            print("ERROR: eval.json file not found")
            print("Usage: python view_eval_results.py [eval.json path]")
            sys.exit(1)
    
    view_eval_results(json_file)

