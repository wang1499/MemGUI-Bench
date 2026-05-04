#!/usr/bin/env python3
import os
import sys
import json
import pandas as pd
import glob
import argparse


def get_success_tasks(results_csv_path):
    df = pd.read_csv(results_csv_path)
    success_col = None
    for c in df.columns:
        if "_direct_with_action_attempt_1_evaluation" in c:
            success_col = c
            break

    if success_col is None:
        return [], [], "No evaluation column found"

    success_tasks = []
    failed_tasks = []
    for _, row in df.iterrows():
        task_id = row["task_identifier"]
        eval_result = str(row.get(success_col, "")).upper().strip()
        if eval_result == "S":
            success_tasks.append(task_id)
        else:
            failed_tasks.append(task_id)

    return success_tasks, failed_tasks, success_col


def get_metrics_summary(session_path):
    metrics_file = os.path.join(session_path, "metrics_summary.json")
    if not os.path.exists(metrics_file):
        return None
    try:
        with open(metrics_file) as f:
            return json.load(f)
    except Exception:
        return None


def get_task_attempts_results(results_dir, session, task_id):
    """Get evaluation results for all attempts of a task."""
    task_dir = os.path.join(results_dir, session, task_id)
    if not os.path.exists(task_dir):
        return {}
    
    results = {}
    # Search for attempt directories under any agent subdirectory
    for agent_dir in os.listdir(task_dir):
        agent_path = os.path.join(task_dir, agent_dir)
        if not os.path.isdir(agent_path):
            continue
        for attempt_dir in sorted(os.listdir(agent_path)):
            if attempt_dir.startswith("attempt_"):
                eval_path = os.path.join(agent_path, attempt_dir, "evaluation_summary.json")
                if os.path.exists(eval_path):
                    try:
                        with open(eval_path) as f:
                            data = json.load(f)
                        attempt_num = int(attempt_dir.split("_")[1])
                        results[attempt_num] = {
                            "result": data.get("final_result", -1),
                            "reason": data.get("final_reason", "")[:100] + "..." if len(data.get("final_reason", "")) > 100 else data.get("final_reason", "")
                        }
                    except Exception:
                        pass
    return results


def get_task_steps(results_dir, session, task_id):
    logs_path = os.path.join(results_dir, session, task_id, "*/attempt_1/detailed_model_logs.json")
    matches = glob.glob(logs_path)
    if not matches:
        return None
    try:
        with open(matches[0]) as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return None


def print_short_success_tasks(results_dir, session, success_tasks, f=None):
    text = f"\n  Success with <10 steps:"
    if f:
        f.write(text + "\n")
    else:
        print(text)
    
    short_tasks = []
    for task in success_tasks:
        steps = get_task_steps(results_dir, session, task)
        if steps is not None and steps < 10:
            short_tasks.append((task, steps))
    
    if short_tasks:
        line = ""
        for i, (task, steps) in enumerate(short_tasks):
            if i % 6 == 0:
                if line:
                    if f:
                        f.write(line + "\n")
                    else:
                        print(line)
                line = "   "
            line += f" {task}({steps})"
        if line:
            if f:
                f.write(line + "\n")
            else:
                print(line)
    else:
        text = "    None"
        if f:
            f.write(text + "\n")
        else:
            print(text)


def main():
    parser = argparse.ArgumentParser(description="Summarize test run success counts")
    parser.add_argument("results_dir", nargs="?", default="results", help="Results directory")
    parser.add_argument("--short", action="store_true", help="Show success tasks with <10 steps")
    parser.add_argument("--output", type=str, help="Output file path (default: scripts/output/success_summary.txt)")
    args = parser.parse_args()
    
    results_dir = args.results_dir
    
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(os.path.dirname(__file__), "output", "success_summary.txt")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    sessions = sorted([
        d for d in os.listdir(results_dir) 
        if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("session-")
    ])
    
    if not sessions:
        print(f"No session directories found in {results_dir}")
        return
    
    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{'Session':<45} {'Success':<10} {'Memory':<10} {'Total':<10} {'Rate'}\n")
        f.write("=" * 80 + "\n")

        all_sessions_data = []

        for session in sessions:
            session_path = os.path.join(results_dir, session)
            csv_path = os.path.join(session_path, "results.csv")
            if not os.path.exists(csv_path):
                f.write(f"{session:<45} {'N/A':<10} (no results.csv)\n")
                continue

            success_tasks, failed_tasks, col_used = get_success_tasks(csv_path)
            total = len(success_tasks) + len(failed_tasks)
            success_count = len(success_tasks)
            rate = f"{success_count/total*100:.1f}%" if total > 0 else "N/A"

            metrics = get_metrics_summary(session_path)
            memory_success = metrics.get("pass_at_1_memory_count", "N/A") if metrics else "N/A"

            f.write(f"{session:<45} {success_count:<10} {memory_success:<10} {total:<10} {rate}\n")

            all_sessions_data.append({
                "session": session,
                "success_count": success_count,
                "total": total,
                "rate": rate,
                "memory_success": memory_success,
                "success_tasks": success_tasks,
                "failed_tasks": failed_tasks,
            })
        
        f.write("=" * 80 + "\n")

        for data in all_sessions_data:
            f.write(f"\n{data['session']} - Success: {data['success_count']}/{data['total']} ({data['rate']}), Memory: {data['memory_success']}\n")
            f.write("-" * 60 + "\n")
            if data["success_tasks"]:
                f.write(f"  Success ({len(data['success_tasks'])}):\n")
                line = ""
                for i, task in enumerate(data["success_tasks"]):
                    if i % 6 == 0:
                        if line:
                            f.write(line + "\n")
                        line = "   "
                    line += f" {task}"
                if line:
                    f.write(line + "\n")
                
                # Show tasks that first succeeded on attempt 2 or 3
                # Need to check failed_tasks too since those have att1=F
                recovered_at_2 = []
                recovered_at_3 = []
                all_tasks = data["success_tasks"] + data["failed_tasks"]
                for task in all_tasks:
                    attempts = get_task_attempts_results(results_dir, data["session"], task)
                    if not attempts:
                        continue
                    att1 = attempts.get(1, {}).get("result", -1)
                    att2 = attempts.get(2, {}).get("result", -1)
                    att3 = attempts.get(3, {}).get("result", -1)
                    # First success at attempt 2: att1=F, att2=S
                    if att1 == 0 and att2 == 1:
                        recovered_at_2.append(task)
                    # First success at attempt 3: att1=F, att2=F, att3=S
                    elif att1 == 0 and att2 == 0 and att3 == 1:
                        recovered_at_3.append(task)
                
                if recovered_at_2:
                    f.write(f"\n  First success at attempt 2 ({len(recovered_at_2)}):\n")
                    line = ""
                    for i, task in enumerate(recovered_at_2):
                        if i % 6 == 0:
                            if line:
                                f.write(line + "\n")
                            line = "   "
                        line += f" {task}"
                    if line:
                        f.write(line + "\n")
                
                if recovered_at_3:
                    f.write(f"\n  First success at attempt 3 ({len(recovered_at_3)}):\n")
                    line = ""
                    for i, task in enumerate(recovered_at_3):
                        if i % 6 == 0:
                            if line:
                                f.write(line + "\n")
                            line = "   "
                        line += f" {task}"
                    if line:
                        f.write(line + "\n")
            
            if args.short:
                print_short_success_tasks(results_dir, data["session"], data["success_tasks"], f)
            
            # if data["failed_tasks"]:
            #     f.write(f"  Failed ({len(data['failed_tasks'])}):\n")
            #     line = ""
            #     for i, task in enumerate(data["failed_tasks"]):
            #         if i % 6 == 0:
            #             if line:
            #                 f.write(line + "\n")
            #             line = "   "
            #         line += f" {task}"
            #     if line:
            #         f.write(line + "\n")
    
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
