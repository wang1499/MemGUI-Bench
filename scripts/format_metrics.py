#!/usr/bin/env python3
import json
import argparse
from pathlib import Path


def format_metrics_summary(metrics: dict) -> str:
    lines = []
    separator = "=" * 100

    lines.append(separator)
    lines.append("\n[>] Task Progress:")
    lines.append(f"    Total: {metrics.get('total_tasks', 0)} tasks (Memory: {metrics.get('memory_tasks', 0)}, Standard: {metrics.get('standard_tasks', 0)})")
    lines.append(f"    Executed: {metrics.get('executed_tasks', 0)}/{metrics.get('total_tasks', 0)} ({metrics.get('executed_tasks', 0) / max(metrics.get('total_tasks', 1), 1) * 100:.1f}%)")
    lines.append(f"    Evaluated: {metrics.get('evaluated_tasks', 0)}/{metrics.get('total_tasks', 0)} ({metrics.get('evaluated_tasks', 0) / max(metrics.get('total_tasks', 1), 1) * 100:.1f}%)")

    lines.append(f"\n[>] Pass@K Results:")

    pass_at_1_count = metrics.get('pass_at_1_count', 0)
    pass_at_1_total = metrics.get('pass_at_1_total', 0)
    pass_at_1_rate = metrics.get('pass_at_1_rate', 0)
    pass_at_2_count = metrics.get('pass_at_2_count', 0)
    pass_at_2_total = metrics.get('pass_at_2_total', 0)
    pass_at_2_rate = metrics.get('pass_at_2_rate', 0)
    pass_at_3_count = metrics.get('pass_at_3_count', 0)
    pass_at_3_total = metrics.get('pass_at_3_total', 0)
    pass_at_3_rate = metrics.get('pass_at_3_rate', 0)
    lines.append(f"    Overall:  @1: {pass_at_1_count}/{pass_at_1_total} ({pass_at_1_rate:.1f}%) | @2: {pass_at_2_count}/{pass_at_2_total} ({pass_at_2_rate:.1f}%) | @3: {pass_at_3_count}/{pass_at_3_total} ({pass_at_3_rate:.1f}%)")

    mem_count = metrics.get('pass_at_1_memory_count', 0)
    mem_total = metrics.get('pass_at_1_memory_total', 0)
    mem_rate = metrics.get('pass_at_1_memory_rate', 0)
    mem2_count = metrics.get('pass_at_2_memory_count', 0)
    mem2_total = metrics.get('pass_at_2_memory_total', 0)
    mem2_rate = metrics.get('pass_at_2_memory_rate', 0)
    mem3_count = metrics.get('pass_at_3_memory_count', 0)
    mem3_total = metrics.get('pass_at_3_memory_total', 0)
    mem3_rate = metrics.get('pass_at_3_memory_rate', 0)
    if mem_total > 0:
        lines.append(f"    Memory:   @1: {mem_count}/{mem_total} ({mem_rate:.1f}%) | @2: {mem2_count}/{mem2_total} ({mem2_rate:.1f}%) | @3: {mem3_count}/{mem3_total} ({mem3_rate:.1f}%)")

    std_count = metrics.get('pass_at_1_standard_count', 0)
    std_total = metrics.get('pass_at_1_standard_total', 0)
    std_rate = metrics.get('pass_at_1_standard_rate', 0)
    std2_count = metrics.get('pass_at_2_standard_count', 0)
    std2_total = metrics.get('pass_at_2_standard_total', 0)
    std2_rate = metrics.get('pass_at_2_standard_rate', 0)
    std3_count = metrics.get('pass_at_3_standard_count', 0)
    std3_total = metrics.get('pass_at_3_standard_total', 0)
    std3_rate = metrics.get('pass_at_3_standard_rate', 0)
    if std_total > 0:
        lines.append(f"    Standard: @1: {std_count}/{std_total} ({std_rate:.1f}%) | @2: {std2_count}/{std2_total} ({std2_rate:.1f}%) | @3: {std3_count}/{std3_total} ({std3_rate:.1f}%)")

    lines.append(f"\n[>] Core Metrics:")
    irr_count = metrics.get('irr_count', 0)
    memory_tasks = metrics.get('memory_tasks', 0)
    avg_irr = metrics.get('avg_irr', 0)
    lines.append(f"    IRR: {avg_irr:.1f}% ({irr_count}/{memory_tasks} memory tasks evaluated)")

    frr = metrics.get('frr', 0)
    r2 = metrics.get('recovery_at_2', 0)
    r3 = metrics.get('recovery_at_3', 0)
    n_failed = metrics.get('n_failed_1', 0)
    lines.append(f"    FRR: {frr:.1f}% (R2={r2}, R3={r3}, first_failures={n_failed})")

    mtpr = metrics.get('mtpr', 0)
    sr_mem = metrics.get('sr_memory_at_1', 0)
    sr_std = metrics.get('sr_standard_at_1', 0)
    lines.append(f"    MTPR: {mtpr:.3f} (Memory@1={sr_mem:.1f}%, Standard@1={sr_std:.1f}%)")

    lines.append(f"\n[>] By Difficulty:")
    diff_keys = sorted([k for k in metrics.keys() if k.startswith('count_diff_')])
    for key in diff_keys:
        diff = key.replace('count_diff_', '')
        count = metrics.get(f'count_diff_{diff}', 0)
        rate_1 = metrics.get(f'pass_at_1_diff_{diff}', 0)
        rate_2 = metrics.get(f'pass_at_2_diff_{diff}', 0)
        rate_3 = metrics.get(f'pass_at_3_diff_{diff}', 0)
        lines.append(f"    D{diff}: @1={rate_1:.1f}% | @2={rate_2:.1f}% | @3={rate_3:.1f}% (n={count})")

    lines.append(f"\n[>] By App Count:")
    apps_keys = sorted([k for k in metrics.keys() if k.startswith('count_apps_')])
    for key in apps_keys:
        apps = key.replace('count_apps_', '')
        count = metrics.get(f'count_apps_{apps}', 0)
        rate_1 = metrics.get(f'pass_at_1_apps_{apps}', 0)
        rate_2 = metrics.get(f'pass_at_2_apps_{apps}', 0)
        rate_3 = metrics.get(f'pass_at_3_apps_{apps}', 0)
        lines.append(f"    {apps} Apps: @1={rate_1:.1f}% | @2={rate_2:.1f}% | @3={rate_3:.1f}% (n={count})")

    lines.append(f"\n[>] Token Statistics:")
    total_prompt = metrics.get('token_att1_total_prompt', 0)
    total_completion = metrics.get('token_att1_total_completion', 0)
    total_steps = metrics.get('token_att1_total_steps', 0)
    avg_prompt = metrics.get('token_att1_avg_prompt', 0)
    avg_completion = metrics.get('token_att1_avg_completion', 0)
    lines.append(f"    Att1: {total_steps} steps | In: {total_prompt:,} (avg {avg_prompt}) | Out: {total_completion:,} (avg {avg_completion})")

    lines.append(separator)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Format metrics_summary.json to terminal output")
    parser.add_argument(
        "--input",
        type=str,
        default="metrics_summary.json",
        help="Path to metrics_summary.json file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (optional, prints to stdout if not specified)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        return 1

    with open(input_path, 'r', encoding='utf-8') as f:
        metrics = json.load(f)

    output = format_metrics_summary(metrics)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output saved to: {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    exit(main())
