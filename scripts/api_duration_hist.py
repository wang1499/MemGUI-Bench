#!/usr/bin/env python3
import json
import os
import sys
import glob
import math
from collections import defaultdict


def load_durations(results_dir):
    durations = []
    patterns = [
        os.path.join(results_dir, "*/Qwen3VL*/attempt_1/detailed_model_logs.json"),
        os.path.join(results_dir, "session-*/Qwen3VL*/attempt_1/detailed_model_logs.json"),
    ]
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            try:
                data = json.load(open(path))
                for entry in data:
                    d = entry.get("api_call_duration", 0) or entry.get("parallel_duration", 0)
                    if d > 0:
                        durations.append(d)
            except Exception:
                continue
    return durations


def histogram(durations, bin_size=1.0):
    if not durations:
        print("No data found.")
        return

    max_val = max(durations)
    min_val = min(durations)
    avg_val = sum(durations) / len(durations)
    median_val = sorted(durations)[len(durations) // 2]

    num_bins = math.ceil(max_val / bin_size) + 1
    bins = [0] * num_bins
    for d in durations:
        idx = int(d / bin_size)
        bins[idx] += 1

    max_count = max(bins) if bins else 1
    bar_width = 50

    task_name = os.path.basename(results_dir.rstrip("/"))
    print(f"API Call Duration Distribution — {task_name}")
    print(f"Total calls: {len(durations)}  |  Min: {min_val:.2f}s  |  Max: {max_val:.2f}s  |  Avg: {avg_val:.2f}s  |  Median: {median_val:.2f}s")
    print("=" * 80)

    for i, count in enumerate(bins):
        if count == 0 and i > 0 and i < num_bins - 1:
            continue
        low = i * bin_size
        high = (i + 1) * bin_size
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        pct = count / len(durations) * 100
        label = f"{low:5.0f}s-{high:5.0f}s"
        print(f"  {label} │{bar:<{bar_width}} {count:>5} ({pct:5.1f}%)")

    print("=" * 80)

    p50 = sorted(durations)[len(durations) // 2]
    p90 = sorted(durations)[int(len(durations) * 0.9)]
    p95 = sorted(durations)[int(len(durations) * 0.95)]
    p99 = sorted(durations)[int(len(durations) * 0.99)] if len(durations) > 1 else p95
    print(f"  P50: {p50:.2f}s  |  P90: {p90:.2f}s  |  P95: {p95:.2f}s  |  P99: {p99:.2f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        results_dir = "/data2/wcl/MemGUI-Bench/results/session-memgui-v26050215-new-owl1"#"/data2/wcl/MemGUI-Bench/results"
    else:
        results_dir = sys.argv[1]
    bin_size = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0

    durations = load_durations(results_dir)
    histogram(durations, bin_size)
