#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Information Retention Rate (IRR) Agent
For calculating the information retention rate of agents during task execution.

This module is used by memgui_eval/evaluator.py as part of the integrated evaluation pipeline.
"""

import os
import sys
import json
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from memgui_eval.utils.llm.llm_api import inference_chat_gemini_wo_image
from memgui_eval.utils.common import parse_json_from_response


def safe_parse_json_from_response(response: str) -> dict:
    """
    Safely parse JSON response, handling format errors
    """
    try:
        return parse_json_from_response(response)
    except Exception as e:
        # Try alternative parsing methods
        try:
            # Remove possible markdown markers
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
            if clean_response.endswith("```"):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()

            return json.loads(clean_response)
        except:
            print(f"    JSON parsing failed: {str(e)[:100]}...")
            return None


def get_irr_analysis_prompt(
    task_description: str, failure_reason: str, step_descriptions: List[Dict]
) -> Tuple[str, str]:
    """
    Generate IRR analysis system and user prompts
    """
    system_prompt = """You are an expert in analyzing agent information retention capabilities. Your task is to precisely calculate the Information Retention Rate (IRR) of an agent based on the given task description, failure reason, and execution step descriptions.

## IRR Definition and Calculation Principles

IRR = (Number of correctly recalled and used information units / Total number of information units required by the task) × 100%

**Information Unit**: The smallest piece of information that the agent is required to remember and use in a task. Examples include:
- Product prices, ratings, specifications
- Contact phone numbers, email addresses
- Meeting dates, times, locations
- Order numbers, verification codes
- Product models, brands, features
- Addresses, rent prices, areas, etc.

## Detailed Calculation Rules

### 1. Task Success
If the task is ultimately successful, it means all required information has been correctly processed.
**IRR = 100%**

### 2. Partial Failure with Explicit Output
Applies to tasks that require explicit output of remembered information (e.g., taking notes, sending messages).
If the task fails but some information units are correctly output, IRR is calculated based on the proportion.
**Example**: Task requires remembering 9 pieces of information, agent correctly outputs 7.
**IRR = 7/9 = 77.8%**

### 3. Failure in Implicit Memory Tasks
Applies to tasks requiring agents to use memory for internal calculations or decisions, ultimately executing only one action.
In such cases, we cannot externally trace the specific correctness of the memory chain.
**For objectivity and consistency, if the final decision behavior is incorrect, IRR = 0%**

**Example**: "Search 6 courses on Coursera, remember each course's rating, review count, and language count, calculate a 'popularity score', and navigate to the highest-scoring course page."
- If the agent navigates to the wrong course page, we cannot determine whether it misremembered ratings, review counts, or made calculation errors.
- Since we cannot objectively assign "partial credit", IRR = 0%.

### 4. Early-Stage Failure
If the agent fails early in the task (e.g., unable to find the information source page), resulting in no information units being processed.
**IRR = 0%**

## Output Format

Your response must be in JSON format containing:
- total_information_units: Total number of information units required (integer)
- correctly_used_units: Number of correctly used information units (integer)
- irr_percentage: IRR percentage (0-100, integer)
- analysis_reason: Detailed analysis reasoning (string)

## Important Notes

- Be precise in counting information units - each specific piece of data counts as one unit
- For implicit memory tasks with wrong final decisions, always assign IRR = 0%
- For explicit output tasks, count the actual correct information in the output
- Consider the task type carefully when applying calculation rules
- Provide clear, objective reasoning for your IRR calculation"""

    # Format step descriptions
    formatted_steps = []
    for i, step_data in enumerate(step_descriptions):
        if isinstance(step_data, dict):
            action = step_data.get("action_description", "N/A")
            ui = step_data.get("ui_description", "N/A")
            formatted_steps.append(
                f"Step {i + 1}:\n  Action: {action}\n  UI State: {ui}"
            )
        else:
            formatted_steps.append(f"Step {i + 1}: {step_data}")

    steps_text = "\n".join(formatted_steps)

    user_prompt = f"""Please analyze the Information Retention Rate (IRR) for the following task:

## Task Description
{task_description}

## Failure Reason
{failure_reason}

## Execution Step Descriptions
{steps_text}

Based on the above information and following the IRR calculation principles, please provide a precise analysis.

Output in JSON format:
```json
{{
  "total_information_units": <integer>,
  "correctly_used_units": <integer>, 
  "irr_percentage": <0-100 integer>,
  "analysis_reason": "<detailed analysis reasoning>"
}}
```"""

    return system_prompt, user_prompt


def calculate_irr_for_task(
    task_description: str,
    failure_reason: str,
    step_descriptions: List[Dict],
    model: str = None,
) -> Optional[Dict]:
    """
    Calculate IRR for a single task

    Args:
        task_description: Description of the task
        failure_reason: Reason why the task failed
        step_descriptions: List of step descriptions from agent execution
        model: LLM model to use for analysis (default: uses config)

    Returns:
        Dictionary containing IRR analysis results or None if failed
    """
    try:
        system_prompt, user_prompt = get_irr_analysis_prompt(
            task_description, failure_reason, step_descriptions
        )

        # Use the model from config if not specified
        from memgui_eval.utils.llm.llm_config import DEFAULT_MODEL

        if model is None:
            model = DEFAULT_MODEL

        response = inference_chat_gemini_wo_image(
            system_prompt, user_prompt, model=model, max_tokens=4096
        )

        if isinstance(response, dict):
            response_str = response["content"]
        else:
            response_str = response

        irr_result = parse_json_from_response(response_str)
        return irr_result

    except Exception as e:
        print(f"Error calculating IRR: {e}")
        return None
