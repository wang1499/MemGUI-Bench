#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path


def find_error_attempts(results_dir):
    error_attempts = []
    results_path = Path(results_dir)

    for eval_summary_path in results_path.rglob("evaluation_summary.json"):
        try:
            with open(eval_summary_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data.get("final_result") == -1:
                attempt_dir = eval_summary_path.parent
                parts = attempt_dir.parts
                if "attempt_" in parts[-1]:
                    attempt_num = int(parts[-1].split("_")[1])
                    agent_name = parts[-2] if len(parts) > 1 else "Unknown"
                    task_id = parts[-3] if len(parts) > 2 else "Unknown"

                    error_attempts.append({
                        "task_identifier": task_id,
                        "agent_name": agent_name,
                        "attempt_num": attempt_num,
                        "attempt_dir": str(attempt_dir),
                        "reason": data.get("final_reason", "Unknown")
                    })
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {eval_summary_path}: {e}")

    return error_attempts


def reevaluate_attempt(task_id, agent_name, attempt_num, results_dir, conda_path="/data/conda_env/MemGUI"):
    memgui_env_python = os.path.join(conda_path, "bin", "python")
    evaluator_script = os.path.join(os.getcwd(), "memgui_eval", "evaluator.py")

    if not os.path.exists(memgui_env_python):
        memgui_env_python = os.path.join(conda_path, "envs", "MemGUI", "bin", "python")

    command = [
        memgui_env_python,
        evaluator_script,
        "--task_identifier", task_id,
        "--result_dir", results_dir,
        "--mode", "full",
        "--agent", agent_name,
        "--attempt_num", str(attempt_num),
    ]

    print(f"Re-evaluating: {task_id} / {agent_name} / attempt_{attempt_num}")
    print(f"Command: {' '.join(command)}")

    import subprocess
    result = subprocess.run(command, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate failed attempts")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to results directory"
    )
    parser.add_argument(
        "--list_errors",
        action="store_true",
        help="List all error attempts without re-evaluating"
    )
    parser.add_argument(
        "--task_id",
        type=str,
        help="Specific task ID to re-evaluate"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Specific agent name to re-evaluate"
    )
    parser.add_argument(
        "--attempt",
        type=int,
        help="Specific attempt number to re-evaluate"
    )
    parser.add_argument(
        "--conda_path",
        type=str,
        default="/data/conda_env/MemGUI",
        help="Path to conda environment"
    )

    args = parser.parse_args()

    if args.task_id and args.agent and args.attempt:
        print(f"Re-evaluating single attempt: {args.task_id} / {args.agent} / attempt_{args.attempt}")
        success = reevaluate_attempt(
            args.task_id,
            args.agent,
            args.attempt,
            args.results_dir,
            args.conda_path
        )
        sys.exit(0 if success else 1)

    error_attempts = find_error_attempts(args.results_dir)

    if args.list_errors:
        print(f"Found {len(error_attempts)} error attempts:\n")
        for i, err in enumerate(error_attempts):
            print(f"{i+1}. Task: {err['task_identifier']}")
            print(f"   Agent: {err['agent_name']}")
            print(f"   Attempt: {err['attempt_num']}")
            print(f"   Reason: {err['reason'][:100]}...")
            print()
        return

    print(f"Found {len(error_attempts)} error attempts. Use --list_errors to see details.")
    print("To re-evaluate a specific attempt, use:")
    print("  python re_evaluate_errors.py --results_dir <path> --task_id <id> --agent <name> --attempt <num>")


if __name__ == "__main__":
    main()
