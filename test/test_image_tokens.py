#!/usr/bin/env python3
"""
测试图片的 token 数
使用 config.yaml 中配置的 qwen3vl 和 gemini-2.5-pro 两个模型
"""

import os
import sys
import base64
from pathlib import Path

# 添加项目根目录到路径
_project_root = os.path.join(os.path.dirname(__file__), "..")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openai import OpenAI

# 从 config.yaml 读取配置
from config_loader import get_config

_config = get_config(verbose=False)

# 图片路径列表（从 api.py 中提取）
IMAGE_PATHS = [
    "results/session-memgui-v26050215-new-owl/001-FindProductAndFilter/Qwen3VL/attempt_1/puzzle/pre_eval_puzzle.png",
    "results/session-memgui-v26050215-new-owl/001-FindProductAndFilter/Qwen3VL/attempt_1/puzzle/puzzle.png",
    "results/session-memgui-v26050215-new-owl/001-FindProductAndFilter/Qwen3VL/attempt_1/visualize_actions/step_1.png",
    "results/session-memgui-v26050215-new-owl-easy/001-FindProductAndFilter/Qwen3VL/attempt_1/0.png",
]


def encode_image(image_path):
    """将图片转为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_model_tokens(model_config, image_paths):
    """
    测试指定模型对图片的 token 数
    model_config: dict with keys: name, base_url, api_key
    """
    print(f"\n{'='*60}")
    print(f"Testing model: {model_config['name']}")
    print(f"Base URL: {model_config['base_url']}")
    print(f"{'='*60}")

    client = OpenAI(base_url=model_config["base_url"], api_key=model_config["api_key"])

    system_prompt = "You are a helpful assistant."
    user_prompt = "Describe what you see in this image briefly."

    for img_path in image_paths:
        full_path = os.path.join(_project_root, img_path)
        if not os.path.exists(full_path):
            print(f"  [SKIP] File not found: {img_path}")
            continue

        try:
            image_base64 = encode_image(full_path)
            completion = client.chat.completions.create(
                model=model_config["name"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,
                temperature=0.01,
            )

            usage = completion.usage
            content = completion.choices[0].message.content
            print(f"  Image: {img_path}")
            print(f"    Prompt tokens: {usage.prompt_tokens}")
            print(f"    Completion tokens: {usage.completion_tokens}")
            print(f"    Total tokens: {usage.total_tokens}")
            print(f"    Sum check: {usage.prompt_tokens + usage.completion_tokens}")
            print(f"    Full usage: {usage}")
            #print(f"    Model output: {content}")

        except Exception as e:
            print(f"  [ERROR] {img_path}: {str(e)}")


def main():
    # 配置两个模型
    models = []

    # 1. Qwen3VL 模型
    qwen_base_url = _config.get("BASE_URL", "http://127.0.0.1:8007/v1")
    qwen_api_key = _config.get("QWEN_API_KEY", "YOUR_API_KEY_HERE")
    qwen_model = _config.get("QWEN_MODEL", "qwen3vl")
    if qwen_api_key:
        models.append({
            "name": qwen_model,
            "base_url": qwen_base_url,
            "api_key": qwen_api_key,
        })
    else:
        print("[WARN] QWEN_API_KEY not set, skipping Qwen3VL test")

    # 2. Gemini 模型 (使用 evaluator 配置)
    gemini_base_url = _config.get("MEMGUI_FINAL_DECISION_BASE_URL")
    gemini_api_key = _config.get("MEMGUI_API_KEY")
    gemini_model = _config.get("MEMGUI_FINAL_DECISION_MODEL", "gemini-2.5-pro")
    if gemini_api_key  and gemini_base_url:
        models.append({
            "name": gemini_model,
            "base_url": gemini_base_url,
            "api_key": gemini_api_key,
        })
    else:
        print("[WARN] MEMGUI_API_KEY or BASE_URL not set, skipping Gemini test")

    if not models:
        print("[ERROR] No valid model configuration found. Please check config.yaml")
        return

    # 测试每个模型
    for model_cfg in models:
        test_model_tokens(model_cfg, IMAGE_PATHS)

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
