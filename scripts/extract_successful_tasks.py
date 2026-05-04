#!/usr/bin/env python3
import os
import sys
import csv
import pandas as pd
from collections import defaultdict


def get_successful_tasks(results_dir):
    successful_tasks = set()
    
    sessions = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("session-")
    ])

    for session in sessions:
        session_path = os.path.join(results_dir, session)
        results_csv = os.path.join(session_path, "results.csv")
        
        if not os.path.exists(results_csv):
            continue

        try:
            df = pd.read_csv(results_csv)
            
            eval_col = None
            for col in df.columns:
                if col.endswith("_evaluation"):
                    eval_col = col
                    break
            
            if eval_col is None:
                continue

            for _, row in df.iterrows():
                task_id = str(row.get("task_identifier", "")).strip()
                eval_result = str(row.get(eval_col, "")).strip().upper()
                if eval_result == "S":
                    successful_tasks.add(task_id)
                    
        except Exception as e:
            print(f"Error reading {results_csv}: {e}")
            continue

    return successful_tasks


def filter_tasks_csv(input_csv, output_csv, successful_tasks):
    if not os.path.exists(input_csv):
        print(f"Input file not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)
    
    original_count = len(df)
    df_filtered = df[df["task_identifier"].isin(successful_tasks)]
    filtered_count = len(df_filtered)
    
    df_filtered.to_csv(output_csv, index=False)
    
    print(f"Original tasks: {original_count}")
    print(f"Successful tasks found: {filtered_count}")
    print(f"Output saved to: {output_csv}")
    
    print(f"\nSuccessful task IDs:")
    for tid in sorted(successful_tasks):
        print(f"  {tid}")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    input_csv = sys.argv[2] if len(sys.argv) > 2 else "datav2/memgui-tasks-easy.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "datav2/memgui-tasks-successv2.csv"

    print(f"Scanning results in: {results_dir}")
    successful_tasks = get_successful_tasks(results_dir)
    print(f"Found {len(successful_tasks)} unique successful tasks\n")
    
    filter_tasks_csv(input_csv, output_csv, successful_tasks)
