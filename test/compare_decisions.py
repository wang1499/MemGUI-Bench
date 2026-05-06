#!/usr/bin/env python3
"""
比较评估两阶段决策：baseline (final_decision_phase1) vs env-aware (final_decision_v3_phase1)
从 prompt_logs.json 中提取两阶段结果进行对比分析
"""

import os
import json
import glob
import argparse
import re
from collections import defaultdict


def extract_decision(llm_response: str) -> dict:
    """从 LLM 原始响应中提取 decision JSON"""
    try:
        text = llm_response
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            return {}
        json_str = text[json_start:json_end]
        data = json.loads(json_str)
        return {
            "decision": int(data.get("decision", -999)),
            "reason": data.get("reason", ""),
            "failure_step": data.get("failure_step"),
            "failure_type": data.get("failure_type", "agent_failure"),
        }
    except Exception:
        return {}


def decision_label(d: int) -> str:
    labels = {
        1: "成功",
        0: "失败(智能体)",
        -1: "冲突/不确定",
        -2: "失败(环境问题)",
    }
    return labels.get(d, f"未知({d})")


def analyze_session(session_path: str):
    """分析单个 session 目录"""
    results = []
    task_ids = set()

    for task_dir in glob.glob(os.path.join(session_path, "*")):
        if not os.path.isdir(task_dir):
            continue
        task_id = os.path.basename(task_dir)

        for model_dir in glob.glob(os.path.join(task_dir, "*")):
            if not os.path.isdir(model_dir):
                continue
            model_name = os.path.basename(model_dir)

            for attempt_dir in glob.glob(os.path.join(model_dir, "attempt_*")):
                prompt_logs_path = os.path.join(attempt_dir, "prompt_logs.json")
                if not os.path.exists(prompt_logs_path):
                    continue

                try:
                    with open(prompt_logs_path, "r", encoding="utf-8") as f:
                        logs = json.load(f)
                except Exception:
                    continue

                baseline_decision = None
                v3_decision = None
                baseline_reason = ""
                v3_reason = ""
                v3_failure_type = "agent_failure"

                for entry in logs:
                    stage = entry.get("stage", "")
                    resp = entry.get("llm_response", "")

                    if stage == "final_decision_phase1":
                        parsed = extract_decision(resp)
                        baseline_decision = parsed.get("decision")
                        baseline_reason = parsed.get("reason", "")

                    elif stage == "final_decision_v3_phase1":
                        parsed = extract_decision(resp)
                        v3_decision = parsed.get("decision")
                        v3_reason = parsed.get("reason", "")
                        v3_failure_type = parsed.get("failure_type", "agent_failure")

                if baseline_decision is None and v3_decision is None:
                    continue

                task_ids.add(task_id)
                results.append({
                    "task_id": task_id,
                    "model": model_name,
                    "attempt": os.path.basename(attempt_dir),
                    "baseline_decision": baseline_decision,
                    "v3_decision": v3_decision,
                    "baseline_reason": baseline_reason[:200],
                    "v3_reason": v3_reason[:200],
                    "v3_failure_type": v3_failure_type,
                    "disagree": baseline_decision != v3_decision,
                })

    return results, task_ids


def generate_report(results: list, task_ids: set, output_path: str = None):
    """生成对比分析报告"""
    total = len(results)
    if total == 0:
        print("没有找到有效的评估结果")
        return

    agree = [r for r in results if not r["disagree"]]
    disagree = [r for r in results if r["disagree"]]

    baseline_only_success = [r for r in results if r["baseline_decision"] == 1]
    baseline_only_fail = [r for r in results if r["baseline_decision"] == 0]
    baseline_only_conflict = [r for r in results if r["baseline_decision"] == -1]

    v3_only_success = [r for r in results if r["v3_decision"] == 1]
    v3_only_fail_env = [r for r in results if r["v3_decision"] == -2]
    v3_only_fail_agent = [r for r in results if r["v3_decision"] == 0]

    lines = []
    lines.append("=" * 70)
    lines.append("两阶段决策对比报告 (Baseline vs Env-Aware v3)")
    lines.append("=" * 70)
    lines.append(f"\n总任务数: {len(task_ids)} 个任务 | {total} 条评估记录")
    lines.append(f"\n【一致性】")
    lines.append(f"  两阶段结论一致: {len(agree)} 条 ({100*len(agree)/total:.1f}%)")
    lines.append(f"  两阶段结论分歧: {len(disagree)} 条 ({100*len(disagree)/total:.1f}%)")

    lines.append(f"\n【Baseline 原始结果分布】")
    lines.append(f"  成功 (decision=1):  {len(baseline_only_success)} 条")
    lines.append(f"  失败 (decision=0):  {len(baseline_only_fail)} 条")
    lines.append(f"  冲突 (decision=-1): {len(baseline_only_conflict)} 条")
    baseline_none = [r for r in results if r["baseline_decision"] is None]
    if baseline_none:
        lines.append(f"  无结果: {len(baseline_none)} 条")

    lines.append(f"\n【v3 环境感知结果分布】")
    lines.append(f"  成功 (decision=1):     {len(v3_only_success)} 条")
    lines.append(f"  失败-智能体 (decision=0):  {len(v3_only_fail_agent)} 条")
    lines.append(f"  失败-环境 (decision=-2):  {len(v3_only_fail_env)} 条")

    env_as_agent = [r for r in disagree if r["baseline_decision"] == 0 and r["v3_decision"] in (1, -2)]
    agent_as_env = [r for r in disagree if r["baseline_decision"] in (1, -1) and r["v3_decision"] == -2]
    other_disagree = [r for r in disagree if r not in env_as_agent and r not in agent_as_env]

    lines.append(f"\n【关键分歧分析】(共 {len(disagree)} 条)")
    if env_as_agent:
        lines.append(f"\n  ◆ Baseline误判为智能体失败，实际为环境问题或成功: {len(env_as_agent)} 条")
        for r in env_as_agent[:5]:
            lines.append(f"    [{r['task_id']}] baseline=0({decision_label(0)}) -> v3={r['v3_decision']}({decision_label(r['v3_decision'])})")
            lines.append(f"      v3理由: {r['v3_reason'][:100]}")

    if agent_as_env:
        lines.append(f"\n  ◆ Baseline判断成功/v3认为环境问题: {len(agent_as_env)} 条")
        for r in agent_as_env[:5]:
            lines.append(f"    [{r['task_id']}] baseline={r['baseline_decision']}({decision_label(r['baseline_decision'])}) -> v3=-2(环境失败)")
            lines.append(f"      v3理由: {r['v3_reason'][:100]}")

    if other_disagree:
        lines.append(f"\n  ◆ 其他分歧: {len(other_disagree)} 条")
        for r in other_disagree[:5]:
            lines.append(f"    [{r['task_id']}] baseline={r['baseline_decision']} -> v3={r['v3_decision']}")

    if v3_only_fail_env:
        lines.append(f"\n【v3识别的环境问题任务】({len(v3_only_fail_env)} 条)")
        for r in v3_only_fail_env[:10]:
            lines.append(f"  [{r['task_id']}] v3_reason: {r['v3_reason'][:120]}")

    lines.append(f"\n【详细分歧列表】")
    for r in disagree:
        lines.append(f"  [{r['task_id']}] {r['model']} {r['attempt']}")
        lines.append(f"    Baseline: {r['baseline_decision']}({decision_label(r['baseline_decision'])}) | {r['baseline_reason'][:80]}")
        lines.append(f"    v3:      {r['v3_decision']}({decision_label(r['v3_decision'])}) | {r['v3_reason'][:80]}")
        lines.append(f"    v3_failure_type: {r['v3_failure_type']}")
        lines.append("")

    report_text = "\n".join(lines)
    print(report_text)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        print(f"\n报告已保存到: {output_path}")

    json_output = output_path.replace(".txt", "_data.json") if output_path else None
    if json_output:
        with open(json_output, "w", encoding="utf-8") as f:
            json.dump({
                "summary": {
                    "total_tasks": len(task_ids),
                    "total_records": total,
                    "agree": len(agree),
                    "disagree": len(disagree),
                    "baseline_success": len(baseline_only_success),
                    "baseline_fail": len(baseline_only_fail),
                    "v3_env_failure": len(v3_only_fail_env),
                },
                "disagree_records": disagree,
                "all_records": results,
            }, f, ensure_ascii=False, indent=2)
        print(f"详细数据已保存到: {json_output}")


def main():
    parser = argparse.ArgumentParser(description="比较 baseline vs v3 环境感知两阶段评估决策")
    parser.add_argument("--path", type=str,
                        default="/data2/wcl/MemGUI-Bench/results/session-memgui-v26050510-new-owl-successv2",
                        help="session 目录路径")
    parser.add_argument("--output", type=str, default=None, help="报告输出路径(.txt)")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        print(f"路径不存在: {args.path}")
        return

    print(f"正在分析: {args.path}")
    results, task_ids = analyze_session(args.path)

    if not results:
        print("未找到任何评估记录，请确认路径是否包含 prompt_logs.json 文件")
        return

    generate_report(results, task_ids, output_path=args.output)


if __name__ == "__main__":
    main()
