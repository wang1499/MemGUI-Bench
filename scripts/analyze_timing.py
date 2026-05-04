#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Timing Statistics Script

Usage:
    python scripts/analyze_timing.py <timing_csv_path>

Example:
    python scripts/analyze_timing.py results/session-memgui-v26042602-debug/timing_records.csv
"""

import sys
import pandas as pd


def analyze_timing(csv_path: str):
    df = pd.read_csv(csv_path)
    
    df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce").fillna(0)
    df["eval_time"] = pd.to_numeric(df["eval_time"], errors="coerce").fillna(0)
    
    df_combined = df.groupby("task_id").agg({
        "inference_time": "sum",
        "eval_time": "sum"
    }).reset_index()
    
    print("=" * 60)
    print("Timing Statistics Summary")
    print("=" * 60)
    
    total_inference = df_combined["inference_time"].sum()
    total_eval = df_combined["eval_time"].sum()
    total_time = total_inference + total_eval
    num_tasks = len(df_combined)
    
    print(f"\n[>] Overall Statistics:")
    print(f"    Total Tasks: {num_tasks}")
    print(f"    Total Inference Time: {total_inference:.2f}s ({total_inference/60:.2f} min)")
    print(f"    Total Eval Time: {total_eval:.2f}s ({total_eval/60:.2f} min)")
    print(f"    Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print(f"\n[>] Average Per Task:")
    print(f"    Avg Inference Time: {total_inference/num_tasks:.2f}s")
    print(f"    Avg Eval Time: {total_eval/num_tasks:.2f}s")
    print(f"    Avg Total Time: {total_time/num_tasks:.2f}s")
    
    print(f"\n[>] Time Distribution:")
    print(f"    Inference: {total_inference/total_time*100:.1f}%")
    print(f"    Eval: {total_eval/total_time*100:.1f}%")
    
    print(f"\n[>] Top 10 Longest Inference Tasks:")
    top_inference = df_combined.nlargest(10, "inference_time")[["task_id", "inference_time"]]
    for _, row in top_inference.iterrows():
        print(f"    {row['task_id']}: {row['inference_time']:.2f}s")
    
    print(f"\n[>] Top 10 Longest Eval Tasks:")
    top_eval = df_combined.nlargest(10, "eval_time")[["task_id", "eval_time"]]
    for _, row in top_eval.iterrows():
        print(f"    {row['task_id']}: {row['eval_time']:.2f}s")
    
    print("=" * 60)


if __name__ == "__main__":

    
    csv_path = "/data2/wcl/MemGUI-Bench/results/session-memgui-v26050315-new-Gemini-v2/timing_records.csv"
    analyze_timing(csv_path)
