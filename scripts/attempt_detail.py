#!/usr/bin/env python3
import os
import sys
import json
import glob
import argparse


def get_task_attempt_status(task_dir, attempt_num):
    """Get the status of a specific attempt - returns (status, result)."""
    attempt_dir = None
    for agent_dir in os.listdir(task_dir):
        agent_path = os.path.join(task_dir, agent_dir)
        if not os.path.isdir(agent_path):
            continue
        cand = os.path.join(agent_path, f"attempt_{attempt_num}")
        if os.path.isdir(cand):
            attempt_dir = cand
            break

    if attempt_dir is None:
        return "not_started", None

    eval_path = os.path.join(attempt_dir, "evaluation_summary.json")
    if os.path.exists(eval_path):
        try:
            with open(eval_path) as f:
                data = json.load(f)
            result = data.get("final_result", -1)
            return "evaluated", result
        except Exception:
            return "error", -1
    else:
        return "executed_no_eval", None


def get_task_attempts_results(task_dir):
    """Get evaluation results for all attempts of a task."""
    if not os.path.exists(task_dir):
        return {}

    results = {}
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
                            "reason": data.get("final_reason", "")[:100]
                        }
                    except Exception:
                        pass
    return results


def print_task_attempt_details(session_dir, f=None):
    """Print detailed attempt results for each task."""
    if not os.path.exists(session_dir):
        msg = f"Directory not found: {session_dir}"
        print(msg)
        return

    tasks = sorted([d for d in os.listdir(session_dir) if os.path.isdir(os.path.join(session_dir, d))])

    if not tasks:
        msg = f"No task directories found in {session_dir}"
        print(msg)
        return

    header = f"{'Task':<30} {'Att1':<10} {'Att2':<10} {'Att3':<10}"
    separator = "-" * 65

    output = [separator, header, separator]

    for task in tasks:
        task_path = os.path.join(session_dir, task)

        status1, result1 = get_task_attempt_status(task_path, 1)
        status2, result2 = get_task_attempt_status(task_path, 2)
        status3, result3 = get_task_attempt_status(task_path, 3)

        def result_str(status, r):
            if status == "not_started":
                return "N/A"
            elif status == "executed_no_eval":
                return "NoEval"
            elif status == "error":
                return "Err"
            elif r == 1:
                return "S"
            elif r == 0:
                return "F"
            elif r == -1:
                return "Err"
            else:
                return "?"

        line = f"{task:<30} {result_str(status1, result1):<10} {result_str(status2, result2):<10} {result_str(status3, result3):<10}"
        output.append(line)

    output.append(separator)

    for line in output:
        if f:
            f.write(line + "\n")
        print(line)

    print(f"\nTotal tasks: {len(tasks)}")
    print("S = Success, F = Fail, N/A = Not started, NoEval = Executed but not evaluated, Err = Error")


def main():
    parser = argparse.ArgumentParser(description="Show task attempt details for a session")
    parser.add_argument("session_dir", nargs="?", default="/data2/wcl/MemGUI-Bench/results", help="Session directory path (default: /data2/wcl/MemGUI-Bench/results)")
    parser.add_argument("--output", "-o", type=str, help="Output file path")
    args = parser.parse_args()

    session_dir = args.session_dir

    # 如果传入的是results目录，找出最新的session
    if session_dir == "/data2/wcl/MemGUI-Bench/results" or session_dir == "results":
        session_dir = "/data2/wcl/MemGUI-Bench/results/session-memgui-v26050315-new-owl"

    if args.output:
        with open(args.output, "w") as f:
            print_task_attempt_details(session_dir, f)
    else:
        print_task_attempt_details(session_dir)

if __name__ == "__main__":
    main()