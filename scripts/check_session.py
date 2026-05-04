#!/usr/bin/env python3
"""
Unified session check script.
Usage: python scripts/check_session.py <session_path> [options]

Options:
  --no-metrics         Skip metrics summary
  --no-short           Skip short success tasks check
  --no-attempt-detail  Skip attempt detail analysis
  --no-api             Skip API duration histogram
  --no-timing          Skip timing analysis
  --short-threshold N  Threshold for short success tasks (default: 10)
  --output DIR         Output directory
"""
import os
import sys
import json
import argparse
import subprocess
import math
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def get_session_name(session_path):
    return Path(session_path).name


def format_metrics_summary(metrics: dict) -> str:
    lines = []
    separator = "=" * 100

    lines.append(separator)
    lines.append("\n[>] Task Progress:")
    lines.append(f"    Total: {metrics.get('total_tasks', 0)} tasks (Memory: {metrics.get('memory_tasks', 0)}, Standard: {metrics.get('standard_tasks', 0)})")
    total = metrics.get('total_tasks', 1)
    lines.append(f"    Executed: {metrics.get('executed_tasks', 0)}/{total} ({metrics.get('executed_tasks', 0) / max(total, 1) * 100:.1f}%)")
    lines.append(f"    Evaluated: {metrics.get('evaluated_tasks', 0)}/{total} ({metrics.get('evaluated_tasks', 0) / max(total, 1) * 100:.1f}%)")

    lines.append(f"\n[>] Pass@K Results:")

    def pass_str(cnt, total, rate):
        return f"@{cnt}/{total} ({rate:.1f}%)"

    lines.append(f"    Overall:  {pass_str(metrics.get('pass_at_1_count', 0), metrics.get('pass_at_1_total', 0), metrics.get('pass_at_1_rate', 0))} | {pass_str(metrics.get('pass_at_2_count', 0), metrics.get('pass_at_2_total', 0), metrics.get('pass_at_2_rate', 0))} | {pass_str(metrics.get('pass_at_3_count', 0), metrics.get('pass_at_3_total', 0), metrics.get('pass_at_3_rate', 0))}")

    mem_total = metrics.get('pass_at_1_memory_total', 0)
    if mem_total > 0:
        lines.append(f"    Memory:   {pass_str(metrics.get('pass_at_1_memory_count', 0), mem_total, metrics.get('pass_at_1_memory_rate', 0))} | {pass_str(metrics.get('pass_at_2_memory_count', 0), metrics.get('pass_at_2_memory_total', 0), metrics.get('pass_at_2_memory_rate', 0))} | {pass_str(metrics.get('pass_at_3_memory_count', 0), metrics.get('pass_at_3_memory_total', 0), metrics.get('pass_at_3_memory_rate', 0))}")

    std_total = metrics.get('pass_at_1_standard_total', 0)
    if std_total > 0:
        lines.append(f"    Standard: {pass_str(metrics.get('pass_at_1_standard_count', 0), std_total, metrics.get('pass_at_1_standard_rate', 0))} | {pass_str(metrics.get('pass_at_2_standard_count', 0), metrics.get('pass_at_2_standard_total', 0), metrics.get('pass_at_2_standard_rate', 0))} | {pass_str(metrics.get('pass_at_3_standard_count', 0), metrics.get('pass_at_3_standard_total', 0), metrics.get('pass_at_3_standard_rate', 0))}")

    lines.append(f"\n[>] Core Metrics:")
    lines.append(f"    IRR: {metrics.get('avg_irr', 0):.1f}% ({metrics.get('irr_count', 0)}/{metrics.get('memory_tasks', 0)} memory tasks evaluated)")
    lines.append(f"    FRR: {metrics.get('frr', 0):.1f}% (R2={metrics.get('recovery_at_2', 0)}, R3={metrics.get('recovery_at_3', 0)}, first_failures={metrics.get('n_failed_1', 0)})")
    lines.append(f"    MTPR: {metrics.get('mtpr', 0):.3f} (Memory@1={metrics.get('sr_memory_at_1', 0):.1f}%, Standard@1={metrics.get('sr_standard_at_1', 0):.1f}%)")

    diff_keys = sorted([k for k in metrics.keys() if k.startswith('count_diff_')])
    if diff_keys:
        lines.append(f"\n[>] By Difficulty:")
        for key in diff_keys:
            diff = key.replace('count_diff_', '')
            count = metrics.get(f'count_diff_{diff}', 0)
            lines.append(f"    D{diff}: @1={metrics.get(f'pass_at_1_diff_{diff}', 0):.1f}% | @2={metrics.get(f'pass_at_2_diff_{diff}', 0):.1f}% | @3={metrics.get(f'pass_at_3_diff_{diff}', 0):.1f}% (n={count})")

    apps_keys = sorted([k for k in metrics.keys() if k.startswith('count_apps_')])
    if apps_keys:
        lines.append(f"\n[>] By App Count:")
        for key in apps_keys:
            apps = key.replace('count_apps_', '')
            count = metrics.get(f'count_apps_{apps}', 0)
            lines.append(f"    {apps} Apps: @1={metrics.get(f'pass_at_1_apps_{apps}', 0):.1f}% | @2={metrics.get(f'pass_at_2_apps_{apps}', 0):.1f}% | @3={metrics.get(f'pass_at_3_apps_{apps}', 0):.1f}% (n={count})")

    lines.append(f"\n[>] Token Statistics:")
    lines.append(f"    Att1: {metrics.get('token_att1_total_steps', 0)} steps | In: {metrics.get('token_att1_total_prompt', 0):,} (avg {metrics.get('token_att1_avg_prompt', 0)}) | Out: {metrics.get('token_att1_total_completion', 0):,} (avg {metrics.get('token_att1_avg_completion', 0)})")

    lines.append(separator)
    return "\n".join(lines)


def find_short_success_tasks(session_path, threshold=10):
    short_tasks = []
    session = Path(session_path)
    for task_dir in session.iterdir():
        if not task_dir.is_dir():
            continue
        for agent_dir in task_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            for attempt_dir in sorted(agent_dir.iterdir()):
                if not attempt_dir.name.startswith('attempt_'):
                    continue
                eval_path = attempt_dir / 'evaluation_summary.json'
                if not eval_path.exists():
                    continue
                try:
                    with open(eval_path) as f:
                        data = json.load(f)
                    if data.get('final_result') == 1:
                        steps = data.get('total_steps')
                        if steps is not None and steps < threshold:
                            short_tasks.append((task_dir.name, agent_dir.name, attempt_dir.name, steps))
                except Exception:
                    continue
    return short_tasks


def run_attempt_detail(session_path, output_file=None):
    script_path = Path(__file__).parent / "attempt_detail.py"
    cmd = [sys.executable, str(script_path), session_path]
    if output_file:
        cmd.extend(["--output", output_file])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode


def run_api_histogram(session_path, output_file=None):
    durations = []
    patterns = [
        "*/Qwen3VL*/attempt_*/detailed_model_logs.json",
    ]
    session = Path(session_path)
    for pattern in patterns:
        for path in sorted(session.glob(pattern)):
            try:
                with open(path) as f:
                    data = json.load(f)
                for entry in data:
                    d = entry.get("api_call_duration", 0) or entry.get("parallel_duration", 0)
                    if d > 0:
                        durations.append(d)
            except Exception:
                continue

    if not durations:
        return "No API duration data found."

    max_val = max(durations)
    min_val = min(durations)
    avg_val = sum(durations) / len(durations)
    sorted_durations = sorted(durations)
    median_val = sorted_durations[len(durations) // 2]
    p50 = sorted_durations[len(durations) // 2]
    p90 = sorted_durations[int(len(durations) * 0.9)]
    p95 = sorted_durations[int(len(durations) * 0.95)]
    p99 = sorted_durations[int(len(durations) * 0.99)] if len(durations) > 1 else p95

    bin_size = 1.0
    num_bins = math.ceil(max_val / bin_size) + 1
    bins = [0] * num_bins
    for d in durations:
        idx = int(d / bin_size)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_width = 50

    task_name = os.path.basename(session_path.rstrip("/"))
    lines = []
    lines.append(f"\n{'=' * 80}")
    lines.append(f"API Call Duration Distribution — {task_name}")
    lines.append(f"Total calls: {len(durations)}  |  Min: {min_val:.2f}s  |  Max: {max_val:.2f}s  |  Avg: {avg_val:.2f}s  |  Median: {median_val:.2f}s")
    lines.append("=" * 80)

    for i, count in enumerate(bins):
        if count == 0 and i > 0 and i < num_bins - 1:
            continue
        low = i * bin_size
        high = (i + 1) * bin_size
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        pct = count / len(durations) * 100
        label = f"{low:5.0f}s-{high:5.0f}s"
        lines.append(f"  {label} │{bar:<{bar_width}} {count:>5} ({pct:5.1f}%)")

    lines.append("=" * 80)
    lines.append(f"  P50: {p50:.2f}s  |  P90: {p90:.2f}s  |  P95: {p95:.2f}s  |  P99: {p99:.2f}s")

    return "\n".join(lines)


def run_timing_analysis(session_path):
    if not HAS_PANDAS:
        return "pandas not installed, skipping timing analysis"

    timing_file = Path(session_path) / "timing_records.csv"
    if not timing_file.exists():
        return "No timing_records.csv found"

    try:
        df = pd.read_csv(timing_file)
        df["inference_time"] = pd.to_numeric(df["inference_time"], errors="coerce").fillna(0)
        df["eval_time"] = pd.to_numeric(df["eval_time"], errors="coerce").fillna(0)

        df_combined = df.groupby("task_id").agg({
            "inference_time": "sum",
            "eval_time": "sum"
        }).reset_index()

        total_inference = df_combined["inference_time"].sum()
        total_eval = df_combined["eval_time"].sum()
        total_time = total_inference + total_eval
        num_tasks = len(df_combined)

        metrics_file = Path(session_path) / "metrics_summary.json"
        total_steps = 0
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            total_steps = metrics.get('token_att1_total_steps', 0)

        avg_inference_per_step = total_inference / total_steps if total_steps > 0 else 0

        lines = []
        lines.append(f"\n{'=' * 60}")
        lines.append("Timing Statistics Summary")
        lines.append("=" * 60)
        lines.append(f"\n[>] Overall Statistics:")
        lines.append(f"    Total Tasks: {num_tasks}")
        lines.append(f"    Total Inference Time: {total_inference:.2f}s ({total_inference/60:.2f} min)")
        lines.append(f"    Total Eval Time: {total_eval:.2f}s ({total_eval/60:.2f} min)")
        lines.append(f"    Total Time: {total_time:.2f}s ({total_time/60:.2f} min)")

        if total_steps > 0:
            lines.append(f"\n[>] Per-Step Statistics:")
            lines.append(f"    Avg Inference Time per Step: {avg_inference_per_step:.2f}s")

        lines.append(f"\n[>] Average Per Task:")
        lines.append(f"    Avg Inference Time: {total_inference/num_tasks:.2f}s")
        lines.append(f"    Avg Eval Time: {total_eval/num_tasks:.2f}s")
        lines.append(f"    Avg Total Time: {total_time/num_tasks:.2f}s")

        lines.append(f"\n[>] Time Distribution:")
        lines.append(f"    Inference: {total_inference/total_time*100:.1f}%")
        lines.append(f"    Eval: {total_eval/total_time*100:.1f}%")
        lines.append("=" * 60)

        return "\n".join(lines)
    except Exception as e:
        return f"Error analyzing timing: {e}"


def main():
    parser = argparse.ArgumentParser(description="Unified session check script")
    parser.add_argument("session_path", nargs="?", default=None, help="Session results directory path")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output directory (default: scripts/output/<session_name>)")
    parser.add_argument("--no-metrics", action="store_true", help="Skip metrics summary")
    parser.add_argument("--no-short", action="store_true", help="Skip short success tasks check")
    parser.add_argument("--no-attempt-detail", action="store_true", help="Skip attempt detail analysis")
    parser.add_argument("--no-api", action="store_true", help="Skip API duration histogram")
    parser.add_argument("--no-timing", action="store_true", help="Skip timing analysis")
    parser.add_argument("--short-threshold", type=int, default=10, help="Threshold for short success tasks (default: 10)")
    args = parser.parse_args()

    if args.session_path:
        session_path = args.session_path
    else:
        session_path = os.getcwd()

    session_path = str(Path(session_path).resolve())
    session_name = get_session_name(session_path)

    if args.output:
        output_dir = args.output
    else:
        script_dir = Path(__file__).parent
        output_dir = script_dir / "output" / session_name

    os.makedirs(output_dir, exist_ok=True)
    print(f"Session: {session_name}")
    print(f"Output directory: {output_dir}")

    output_lines = []
    output_lines.append(f"Session: {session_name}")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append("")

    if not args.no_metrics:
        metrics_file = Path(session_path) / "metrics_summary.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            formatted = format_metrics_summary(metrics)
            print(formatted)
            output_lines.append(formatted)

    if not args.no_short:
        short_tasks = find_short_success_tasks(session_path, args.short_threshold)
        short_header = f"\n[>] Success with <{args.short_threshold} steps:"
        print(short_header)
        output_lines.append(short_header)
        if short_tasks:
            for task, agent, attempt, steps in sorted(short_tasks):
                line = f"   {task} | {agent} | {attempt} | {steps} steps"
                print(line)
                output_lines.append(line)
        else:
            line = "   None"
            print(line)
            output_lines.append(line)

    if not args.no_attempt_detail:
        attempt_output_file = Path(output_dir) / "attempt_detail.txt"
        stdout, stderr, rc = run_attempt_detail(session_path, str(attempt_output_file))
        if stdout:
            print(stdout)
            output_lines.append(stdout)
        if stderr and rc != 0:
            print(f"Attempt detail error: {stderr}", file=sys.stderr)

    if not args.no_api:
        api_output = run_api_histogram(session_path)
        print(api_output)
        output_lines.append(api_output)
        api_output_file = Path(output_dir) / "api_duration_hist.txt"
        with open(api_output_file, 'w') as f:
            f.write(api_output)

    if not args.no_timing:
        timing_output = run_timing_analysis(session_path)
        print(timing_output)
        output_lines.append(timing_output)
        timing_output_file = Path(output_dir) / "timing_summary.txt"
        with open(timing_output_file, 'w') as f:
            f.write(timing_output)

    output_file = Path(output_dir) / "session_check.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(output_lines))
    print(f"\nOutput saved to: {output_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
