#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BadCase Agent
For analyzing and classifying failed task execution cases with detailed memory hallucination analysis
"""

import os
import sys
import json
import pandas as pd
import argparse
from typing import Dict, List, Tuple, Optional

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from memgui_eval.utils.llm.llm_api import inference_chat_gemini_wo_image
from memgui_eval.utils.common import parse_json_from_response
import re


def clean_category_value(category: str) -> str:
    """
    Clean and standardize category values
    
    Args:
        category: Original category string
        
    Returns:
        Cleaned category string
    """
    if not isinstance(category, str):
        return str(category)
    
    # Remove leading/trailing whitespace and special characters
    category = category.strip()
    
    # Remove possible punctuation and extra characters
    category = re.sub(r'[^\w\u4e00-\u9fff_]', '', category)
    
    # Define mapping rules to handle common variants
    mapping_rules = {
        # Process memory hallucination related
        r'.*process.*memory.*hallucination.*': "process_memory_hallucination",
        r'.*process.*type.*memory.*': "process_memory_hallucination",
        r'.*流程.*全记忆幻觉.*': "process_memory_hallucination",
        r'.*全记忆幻觉.*流程.*': "process_memory_hallucination",
        r'流程型.*': "process_memory_hallucination",
        
        # Output memory hallucination (formerly memory_memory_hallucination)
        r'.*output.*memory.*hallucination.*': "output_memory_hallucination",
        r'.*memory.*type.*memory.*hallucination.*': "output_memory_hallucination",
        r'.*memory.*memory.*': "output_memory_hallucination",
        r'.*记忆.*全记忆幻觉.*': "output_memory_hallucination",
        r'.*全记忆幻觉.*记忆.*': "output_memory_hallucination",
        r'记忆型.*': "output_memory_hallucination",
        
        # Partial memory hallucination (based on IRR)
        r'.*partial.*memory.*hallucination.*': "partial_memory_hallucination",
        r'.*部分.*记忆.*幻觉.*': "partial_memory_hallucination",
        
        # Knowledge deficiency (formerly agent_knowledge)
        r'.*knowledge.*deficiency.*': "knowledge_deficiency",
        r'.*agent.*knowledge.*': "knowledge_deficiency",
        r'.*knowledge.*': "knowledge_deficiency",
        
        # Intent understanding
        r'.*intent.*understand.*': "intent_understanding",
        r'.*understand.*': "intent_understanding",
        r'.*意图.*理解.*': "intent_understanding",
        r'.*理解.*': "intent_understanding",
        
        # Other
        r'.*other.*': "other",
        r'.*其他.*': "other",
    }
    
    # Apply mapping rules
    for pattern, target in mapping_rules.items():
        if re.match(pattern, category, re.IGNORECASE):
            return target
    
    # If no rule matches, return original value
    return category


def classify_by_irr(irr_percentage: float) -> Optional[str]:
    """
    Classify failure case based on IRR (Information Retention Rate)
    
    Args:
        irr_percentage: IRR percentage value (0-100)
        
    Returns:
        Category string if can be determined by IRR, None otherwise
    """
    if irr_percentage is None or not isinstance(irr_percentage, (int, float)):
        return None
    
    # IRR between 0-100% (exclusive) indicates partial memory hallucination
    if 0 < irr_percentage < 100:
        return "partial_memory_hallucination"
    
    # IRR = 0 or IRR = 100 needs LLM analysis
    return None


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
            if clean_response.startswith('```json'):
                clean_response = clean_response[7:]
            if clean_response.endswith('```'):
                clean_response = clean_response[:-3]
            clean_response = clean_response.strip()
            
            return json.loads(clean_response)
        except:
            print(f"    JSON parsing failed: {str(e)[:100]}...")
            return None


def get_bad_case_analysis_prompt(task_description: str, failure_reason: str, step_descriptions: List[Dict], irr_info: Dict = None) -> Tuple[str, str]:
    """
    Generate BadCase analysis system and user prompts
    """
    system_prompt = """You are an expert in analyzing agent task failure causes. Your task is to precisely analyze the root cause of agent failures in GUI automation tasks and classify them into predefined categories.

## Analysis Principles and Classification System

You need to classify failure cases into one of the following categories:

### 1. Complete Memory Hallucination (Two subtypes)

#### 1a. Process-Type Memory Hallucination (process_memory_hallucination)
Agent forgets what it is doing during task execution, characterized by:
- Suddenly "loses track" and starts clicking on irrelevant pages in loops
- Repeatedly performs unrelated operations, deviating from the main task
- Completely loses direction midway through the task
- Falls into meaningless operation loops

#### 1b. Output Memory Hallucination (output_memory_hallucination)
Agent completes the process correctly in memory-intensive tasks but hallucinates when "submitting the answer":
- Forgets or incorrectly records previously collected information when saving to notes
- Sends wrong content when sending answers via SMS
- Forgets key information at the final output stage
- Collects information correctly but outputs incorrectly

### 2. Knowledge Deficiency (knowledge_deficiency)
Agent lacks basic knowledge or skills required to complete the task:
- Doesn't know how to use specific applications (e.g., doesn't know how to use a check-in app)
- Doesn't understand correct methods for certain UI operations
- Lacks basic understanding of application functionality
- Technical capabilities insufficient to meet task requirements

### 3. Intent Understanding Issues (intent_understanding)
Agent doesn't correctly understand the user's intent in the task description:
- Misunderstands task goals
- Ignores key requirements in the task
- Performs operations unrelated to the task
- Deviates in understanding task instructions

### 4. Other Reasons (other)
If none of the above categories apply, please describe the specific reason for the agent's failure in detail.

**Note**: "partial_memory_hallucination" is automatically determined by IRR analysis and should NOT be used in LLM classification.

## Important Analysis Principles

1. **Prioritize Memory Issues**: Focus on whether failure is due to "forgetting" previous operations
2. **Distinguish Process vs Output Type**: 
   - Process-type: Loses direction midway
   - Output-type: Correct process but wrong final output
3. **Objective Analysis**: Evidence-based analysis using specific execution steps
4. **Be Willing to Say "Other"**: If it truly doesn't fit known categories, don't force classification

## Strict Output Format Requirements

Your response must be in standard JSON format, with the category field being one of the following five values (strict match, no other variants allowed):

**Allowed category values (must match exactly):**
1. "process_memory_hallucination" - Agent forgets goal midway, starts looping or unrelated operations
2. "output_memory_hallucination" - Agent executes process correctly but makes memory errors at final output
3. "knowledge_deficiency" - Agent doesn't know how to use an application or feature
4. "intent_understanding" - Agent misunderstands task requirements
5. "other" - Specific reason description when none of the above fit

**JSON Field Requirements:**
- category: Must be one of the above five values (string)
- confidence: Confidence score, float between 0.0-1.0
- analysis_reason: Detailed analysis process (string)
- key_failure_point: Key failure point (string)
- evidence: Specific evidence supporting the classification (string)
- suggested_improvement: Improvement suggestions (string)

## Analysis Steps

1. **Carefully read** the task description and understand the expected goal
2. **Analyze step-by-step** the execution process and identify key turning points
3. **Determine failure type**: Is it a memory issue, capability issue, understanding issue, or other?
4. **If it's a memory issue**: Distinguish whether it's process-type or output-type
5. **Provide specific evidence** to support your classification decision"""

    # Format step descriptions
    formatted_steps = []
    for i, step_data in enumerate(step_descriptions):
        if isinstance(step_data, dict):
            action = step_data.get("action_description", "N/A")
            ui = step_data.get("ui_description", "N/A") 
            formatted_steps.append(f"Step {i+1}:\n  Action: {action}\n  UI State: {ui}")
        else:
            formatted_steps.append(f"Step {i+1}: {step_data}")
    
    steps_text = "\n".join(formatted_steps)

    # Add IRR information if available
    irr_text = ""
    if irr_info:
        irr_text = f"""
## IRR Analysis Reference
- IRR Percentage: {irr_info.get('irr_percentage', 'N/A')}%
- Total Information Units: {irr_info.get('total_units', 'N/A')}
- Correctly Used Units: {irr_info.get('correct_units', 'N/A')}
- IRR Analysis Reason: {irr_info.get('irr_reason', 'N/A')}

Note: Current case has specific IRR value, requiring detailed failure analysis.
"""

    user_prompt = f"""Please analyze the root cause of the following task failure case:

## Task Description
{task_description}

## Failure Reason
{failure_reason}

## Execution Step Descriptions
{steps_text}

{irr_text}

Please conduct an in-depth analysis following these steps:

1. **Understand Task Goal**: What does this task require the agent to do?
2. **Track Execution Process**: What did the agent actually do? Where did it start deviating?
3. **Identify Failure Type**:
   - If it's a memory issue, is it mid-process confusion (process-type) or final output error (memory-type)?
   - If not a memory issue, is it insufficient capability or understanding error?
4. **Provide Specific Evidence**: Which steps support your judgment?
5. **Give Improvement Suggestions**: How to avoid similar failures?

## Strict Classification Guidelines:
- If agent starts looping clicks or performing unrelated operations midway → "process_memory_hallucination"
- If agent executes correctly but outputs wrong information at the end → "output_memory_hallucination"  
- If agent doesn't know how to use an application → "knowledge_deficiency"
- If agent misunderstands task requirements → "intent_understanding"
- If none of the above fit, describe specific reason in detail → "other"

**Important: The category field must exactly match one of the following five values:**
- "process_memory_hallucination"
- "output_memory_hallucination"
- "knowledge_deficiency"
- "intent_understanding"
- "other"

Output JSON format (strictly follow this format):
```json
{{
  "category": "must be one of the above five values",
  "confidence": 0.95,
  "analysis_reason": "detailed analysis process",
  "key_failure_point": "key failure point",
  "evidence": "specific evidence supporting the classification",
  "suggested_improvement": "improvement suggestions"
}}
```"""

    return system_prompt, user_prompt


def calculate_bad_case_for_task(
    task_description: str,
    failure_reason: str, 
    step_descriptions: List[Dict],
    irr_info: Dict = None,
    model: str = None,
    provider: str = None,
    api_url: str = None,
) -> Optional[Dict]:
    """
    Calculate BadCase classification for a single task
    
    Args:
        task_description: Description of the task
        failure_reason: Reason why the task failed
        step_descriptions: List of step descriptions from agent execution
        irr_info: IRR analysis information for reference
        model: LLM model to use for analysis (defaults to DEFAULT_MODEL from config)
        provider: LLM provider (defaults to DEFAULT_PROVIDER from config)
        api_url: API URL (defaults to DEFAULT_API_URL from config)
        
    Returns:
        Dictionary containing BadCase analysis results or None if failed
    """
    try:
        system_prompt, user_prompt = get_bad_case_analysis_prompt(
            task_description, failure_reason, step_descriptions, irr_info
        )
        
        response = inference_chat_gemini_wo_image(
            system_prompt, user_prompt, model=model, provider=provider, api_url=api_url, max_tokens=4096
        )
        
        if isinstance(response, dict):
            response_str = response["content"]
        else:
            response_str = response
            
        bad_case_result = parse_json_from_response(response_str)
        
        # 验证返回结果的格式
        if bad_case_result and isinstance(bad_case_result, dict):
            # 确保所有必需的键都存在
            required_keys = ['category', 'confidence', 'analysis_reason', 'key_failure_point', 'evidence', 'suggested_improvement']
            for key in required_keys:
                if key not in bad_case_result:
                    bad_case_result[key] = 'N/A'
            
            # 确保confidence是数字
            try:
                bad_case_result['confidence'] = float(bad_case_result['confidence'])
            except (ValueError, TypeError):
                bad_case_result['confidence'] = 0.0
            
            # 确保所有文本字段都是字符串
            for key in ['category', 'analysis_reason', 'key_failure_point', 'evidence', 'suggested_improvement']:
                if not isinstance(bad_case_result[key], str):
                    bad_case_result[key] = str(bad_case_result[key])
            
            # Strict validation and cleaning of category field
            valid_categories = [
                "process_memory_hallucination",
                "output_memory_hallucination",
                "partial_memory_hallucination",  # Set by IRR analysis, not LLM
                "knowledge_deficiency",
                "intent_understanding",
                "other"
            ]
            
            original_category = bad_case_result['category'].strip()
            
            # Direct match
            if original_category in valid_categories:
                bad_case_result['category'] = original_category
            else:
                # Try fuzzy matching and cleaning
                cleaned_category = clean_category_value(original_category)
                if cleaned_category in valid_categories:
                    bad_case_result['category'] = cleaned_category
                    print(f"    Warning: Category cleaned: '{original_category}' -> '{cleaned_category}'")
                else:
                    # If still no match, set to "other"
                    bad_case_result['category'] = "other"
                    print(f"    Error: Category unrecognized: '{original_category}' -> 'other'")
                    # Add original classification info to analysis_reason
                    bad_case_result['analysis_reason'] = f"Original classification: {original_category}\n{bad_case_result['analysis_reason']}"
        
        return bad_case_result
        
    except Exception as e:
        print(f"Error calculating BadCase: {e}")
        return None


def get_agent_name_from_results_csv(csv_path: str) -> str:
    """
    Extract agent name from results.csv file column names
    
    Args:
        csv_path: Path to the results.csv file
        
    Returns:
        Extracted agent name or "unknown_agent" if not found
    """
    df = pd.read_csv(csv_path, nrows=1)
    columns = df.columns.tolist()
    
    # Find columns containing _evaluation and extract agent name
    for col in columns:
        if '_evaluation' in col and 'attempt_1' in col:
            # Example: M3A_vivo_gemini_direct_with_action_attempt_1_evaluation
            # Extract: M3A_vivo_gemini
            parts = col.split('_')
            # Find the position of direct_with_action, agent name is before it
            if 'direct' in parts and 'with' in parts and 'action' in parts:
                direct_idx = parts.index('direct')
                agent_name = '_'.join(parts[:direct_idx])
                return agent_name
    
    return "unknown_agent"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BadCase classification for MemGUI evaluation results")
    parser.add_argument("--agent_dir", type=str, help="Path to specific agent directory")
    parser.add_argument("--all_agents", action="store_true", help="Process all agents in results/00_baselines")
    
    args = parser.parse_args()
    
    if args.all_agents:
        print("BadCase analysis for all agents - use bad_case_parallel_processor.py")
    elif args.agent_dir:
        print(f"BadCase analysis for single agent: {args.agent_dir}")
    else:
        print("Please specify --agent_dir or --all_agents")
        print("Usage examples:")
        print("  python3 bad_case_agent.py --agent_dir ../../results/00_baselines/250721_M3A_gemini-2.5-pro")
        print("  python3 bad_case_agent.py --all_agents")
