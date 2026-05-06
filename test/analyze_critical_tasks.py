#!/usr/bin/env python3
"""
使用 Gemini 模型辅助分析关键任务
重点关注：
1. 网络/环境问题导致的失败（可能需要重试）
2. 成功但描述可疑的任务（可能误判）
3. 其他异常情况
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_project_root = os.path.join(os.path.dirname(__file__), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openai import OpenAI
from config_loader import get_config

_config = get_config(verbose=False)

# Gemini 配置
GEMINI_BASE_URL = _config.get("MEMGUI_FINAL_DECISION_BASE_URL", "http://10.63.48.150:8001/v1")
GEMINI_API_KEY = _config.get("MEMGUI_API_KEY", "YOUR_API_KEY_HERE")
GEMINI_MODEL = "gemini-2.5-pro"  # 或 gemini-3-pro-preview-new-priority

client = OpenAI(base_url=GEMINI_BASE_URL, api_key=GEMINI_API_KEY)


def analyze_task_with_gemini(task_data):
    """使用 Gemini 分析单个任务是否需要重点关注"""
    tid = task_data.get("task_identifier", "unknown")
    task_desc = task_data.get("task_description", "")
    final_result = task_data.get("final_result", -1)
    final_reason = task_data.get("final_reason", "")
    failure_step = task_data.get("failure_step", None)

    system_prompt = """You are an expert task evaluator. Analyze the given task evaluation and determine if it needs human review.

Respond with a JSON object with these exact fields:
{
    "needs_review": true/false,
    "priority": "high/medium/low",
    "category": "success/partial_success/wrong_evaluation/network_issue/app_not_found/other",
    "explanation": "brief explanation in English",
    "suggested_action": "retry/reevaluate/ignore/check_manually"
}

Category definitions:
- success: Task completed correctly, no issues
- partial_success: Task completed but with minor issues or incomplete
- wrong_evaluation: Evaluation result contradicts the evidence (e.g., marked success but reason shows failure, or vice versa)
- network_issue: Failure due to network connectivity problems
- app_not_found: Failure due to app not available or not installed
- other: Other reasons for failure or review needed

For successful tasks (final_result=1): consider if it was truly successful or a false positive.
For failed tasks (final_result=0): determine the actual cause and if evaluation was fair."""

    user_prompt = f"""Task ID: {tid}
Task Description: {task_desc}
Final Result: {"Success" if final_result == 1 else "Failure" if final_result == 0 else "Conflict"}
Final Reason: {final_reason}
Failure Step: {failure_step}

Analyze if this task needs human review. Consider:
1. Was the failure due to environment issues (network, app not available) rather than agent capability?
2. Was a success actually a partial success or false positive?
3. Are there obvious evaluation errors?"""

    try:
        completion = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=5000,
            temperature=0.01,
        )

        content = completion.choices[0].message.content
        # 尝试从文本中提取 JSON（处理 markdown 代码块）
        try:
            # 移除 markdown 代码块标记
            content_clean = content.strip()
            if content_clean.startswith('```json'):
                content_clean = content_clean[7:]
            elif content_clean.startswith('```'):
                content_clean = content_clean[3:]
            if content_clean.endswith('```'):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()
            
            # 查找 JSON 块
            start = content_clean.find('{')
            end = content_clean.rfind('}') + 1
            if start >= 0 and end > start:
                analysis = json.loads(content_clean[start:end])
            else:
                analysis = json.loads(content_clean)
        except json.JSONDecodeError as e:
            # 如果解析失败，创建一个默认结果
            analysis = {
                "needs_review": True,
                "priority": "medium",
                "category": "parse_error",
                "explanation": f"无法解析模型输出: {str(e)[:50]}, content={content[:150]}",
                "suggested_action": "check_manually",
            }
        analysis["task_id"] = tid
        analysis["final_result"] = final_result
        return analysis

    except Exception as e:
        return {
            "task_id": tid,
            "final_result": final_result,
            "needs_review": True,
            "priority": "medium",
            "category": "analysis_error",
            "explanation": f"分析出错: {str(e)}",
            "suggested_action": "check_manually",
        }


def main(path, output_path="/tmp/gemini_analysis.json"):
    import glob

    # 从 path 查找所有 evaluation_summary.json
    summary_files = glob.glob(os.path.join(path, "**/evaluation_summary.json"), recursive=True)
    summary_files.sort()

    # 解析所有任务
    all_tasks = []
    for fp in summary_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_id = data.get("task_identifier", os.path.basename(os.path.dirname(fp)))
        all_tasks.append({
            "task_identifier": task_id,
            "task_description": data.get("task_description", ""),
            "final_result": data.get("final_result", -1),
            "final_reason": data.get("final_reason", ""),
            "failure_step": data.get("failure_step", None),
        })

    print(f"总共 {len(all_tasks)} 个任务需要分析")

    # 所有任务都需要分析
    tasks_to_analyze = all_tasks

    print(f"筛选出 {len(tasks_to_analyze)} 个需要分析的任务")

    # 多线程分析
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_task_with_gemini, t): t for t in tasks_to_analyze}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(tasks_to_analyze)}")

    # 分类统计
    needs_review = [r for r in results if r.get("needs_review")]
    by_priority = {"high": [], "medium": [], "low": []}
    for r in needs_review:
        p = r.get("priority", "medium")
        by_priority[p].append(r)

    by_category = {}
    for r in needs_review:
        cat = r.get("category", "other")
        by_category[cat] = by_category.get(cat, [])
        by_category[cat].append(r)

    # 按原始结果统计
    original_success = [r for r in results if r.get("final_result") == 1]
    original_failure = [r for r in results if r.get("final_result") == 0]
    success_need_review = [r for r in results if r.get("final_result") == 1 and r.get("needs_review")]
    failure_need_review = [r for r in results if r.get("final_result") == 0 and r.get("needs_review")]

    # 生成报告文本
    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("Gemini 辅助分析报告")
    report_lines.append("=" * 60)

    report_lines.append(f"\n总任务数: {len(results)} 个")
    report_lines.append(f"  原始成功: {len(original_success)} 个 | 成功需复查: {len(success_need_review)} 个")
    report_lines.append(f"  原始失败: {len(original_failure)} 个 | 失败需复查: {len(failure_need_review)} 个")

    report_lines.append(f"\n需要人工复核的任务: {len(needs_review)} 个")
    report_lines.append(f"  高优先级: {len(by_priority['high'])} 个")
    report_lines.append(f"  中优先级: {len(by_priority['medium'])} 个")
    report_lines.append(f"  低优先级: {len(by_priority['low'])} 个")

    report_lines.append(f"\n按类别分布:")
    for cat, items in sorted(by_category.items(), key=lambda x: -len(x[1])):
        report_lines.append(f"  {cat}: {len(items)} 个")

    report_lines.append(f"\n高优先级任务 (需要立即关注):")
    for r in by_priority["high"]:
        report_lines.append(f"  [{r['task_id']}] {r.get('category', 'unknown')}: {r.get('explanation', '')}")
        report_lines.append(f"    建议: {r.get('suggested_action', '')}")

    report_lines.append(f"\n中优先级任务 (建议检查):")
    for r in by_priority["medium"][:10]:
        report_lines.append(f"  [{r['task_id']}] {r.get('category', 'unknown')}: {r.get('explanation', '')}")

    report_text = "\n".join(report_lines)
    print(report_text)

    # 保存完整结果
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    json_path = output_path if output_path.endswith(".json") else output_path + ".json"
    txt_path = json_path.replace(".json", "_report.txt")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n完整分析结果已保存到 {json_path}")
    print(f"报告文本已保存到 {txt_path}")


if __name__ == "__main__":
    path = "/data2/wcl/MemGUI-Bench/results/session-memgui-v26050416-new-owl"
    default_output = os.path.join(os.path.dirname(__file__), os.path.basename(os.path.normpath(path)) + "/gemini_analysis.json")
    output_path = default_output
    main(path=path, output_path=output_path)
