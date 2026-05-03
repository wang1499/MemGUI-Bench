#!/usr/bin/env python3
import os
import sys
import json
import glob
from collections import defaultdict


def analyze_tool_usage(results_dir):
    sessions = sorted([
        d for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("session-")
    ])

    if not sessions:
        print(f"No session directories found in {results_dir}")
        return

    print("=" * 90)
    print(f"{'Session':<45} {'mobile_use':<12} {'write_todos':<12} {'write_memory':<12}")
    print("=" * 90)

    for session in sessions:
        session_path = os.path.join(results_dir, session)
        task_stats_files = sorted(glob.glob(os.path.join(session_path, "*/Qwen3VL*/attempt_1/model_call_stats.json")))
        
        if not task_stats_files:
            continue

        session_totals = {"mobile_use": 0, "write_todos": 0, "write_memory": 0}
        
        for stats_path in task_stats_files:
            try:
                with open(stats_path) as f:
                    data = json.load(f)
                tool_stats = data.get("tool_call_stats", {})
                if tool_stats:
                    session_totals["mobile_use"] += tool_stats.get("mobile_use", 0)
                    session_totals["write_todos"] += tool_stats.get("write_todos", 0)
                    session_totals["write_memory"] += tool_stats.get("write_memory", 0)
            except Exception:
                pass

        print(f"{session:<45} {session_totals['mobile_use']:<12} {session_totals['write_todos']:<12} {session_totals['write_memory']:<12}")

    print("=" * 90)

    print("\n" + "=" * 90)
    print("Per-Task Tool Usage Details")
    print("=" * 90)

    for session in sessions:
        session_path = os.path.join(results_dir, session)
        task_dirs = sorted(glob.glob(os.path.join(session_path, "*/Qwen3VL*/attempt_1/model_call_stats.json")))
        if not task_dirs:
            continue

        print(f"\n{session}:")
        session_totals = {"mobile_use": 0, "write_todos": 0, "write_memory": 0}

        for task_path in task_dirs[:10]:
            task_name = task_path.split("/")[-4]
            try:
                with open(task_path) as f:
                    data = json.load(f)
                tool_stats = data.get("tool_call_stats", {})
                if tool_stats:
                    mobile = tool_stats.get("mobile_use", 0)
                    todos = tool_stats.get("write_todos", 0)
                    memory = tool_stats.get("write_memory", 0)
                    print(f"  {task_name:<40} mobile={mobile:3d}  todos={todos:3d}  memory={memory:3d}")
                    session_totals["mobile_use"] += mobile
                    session_totals["write_todos"] += todos
                    session_totals["write_memory"] += memory
            except Exception:
                pass

        if len(task_dirs) > 10:
            print(f"  ... and {len(task_dirs) - 10} more tasks")

        print(f"  {'Session Total':<40} mobile={session_totals['mobile_use']:3d}  todos={session_totals['write_todos']:3d}  memory={session_totals['write_memory']:3d}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results_dir = "results"
    else:
        results_dir = sys.argv[1]

    analyze_tool_usage(results_dir)
