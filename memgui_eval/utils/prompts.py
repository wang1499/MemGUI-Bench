def get_describe_step_prompt(
    task_description: str, log_action: str, log_detail: str
 ) -> (str, str):
    """
    Returns the system and user prompts for describing a single step/image in JSON format.
    """
    system_prompt = (
        "You are an expert mobile device assistant. Your task is to analyze a two-panel image showing the 'Before Action' and 'After Action' state of a user's workflow. "
        "Your analysis must focus *only* on the 'Before Action' panel (the left side). "
        "You must output your response in a JSON format."
    )
    user_prompt = (
        f"The overall task is: '{task_description}'.\n\n"
        "## Input Analysis\n"
        "The provided image shows a 'Before Action' state on the left and an 'After Action' state on the right. "
        "Your entire analysis should focus on the left 'Before Action' panel.\n\n"
        "**Note:** If the 'After Action' panel is identical to the 'Before Action' panel, it signifies this is the final action in the task.\n\n"
        "On the left panel, a user action is visualized with markers: a red circle shows the click/touch point, surrounded by a green square, with a 'C' label in the corner. "
        "The raw action from the execution log is provided for context:\n"
        f"- Action Type: `{log_action}`\n"
        f"- Action Detail: `{log_detail}`\n\n"
        "## Your Task\n"
        "Based on the visual evidence in the **left panel** and the provided log context, perform the following two tasks:\n"
        "1. **action_description**: In your own words, crisply describe the specific action performed (e.g., 'Clicked the \"Settings\" button', 'Typed \"hello\" into the search bar').\n"
        "2. **ui_description**: List the key UI elements visible *in the left panel* that are relevant to the action and the overall task. Do not mention the panel name (e.g., 'Before Action') in your description.\n\n"
        "Your output MUST be a JSON object with these two keys.\n\n"
        "### Example\n"
        "```json\n"
        "{{\n"
        '  "action_description": "The user clicked on the settings icon at the bottom of the screen.",\n'
        '  "ui_description": "The home screen with various app icons is visible. Key elements include the Phone, Messages, and Settings icons at the bottom."\n'
        "}}\n"
        "```"
    )
    return system_prompt, user_prompt


def get_describe_final_step_prompt(
    task_description: str, log_action: str, log_detail: str
 ) -> (str, str):
    """
    Returns the system and user prompts for describing the final single-step image.
    """
    system_prompt = (
        "You are an expert mobile device assistant. Your task is to analyze a single image showing the state of a screen after a user's action. "
        "The user's action (e.g., a click) is visualized on the image itself. "
        "Your analysis must focus on the provided image and the action context. You must output your response in a JSON format."
    )
    user_prompt = (
        f"The overall task is: '{task_description}'.\n\n"
        "## Input Analysis\n"
        "The provided image shows the state of the screen after a final user action. "
        "On the image, the user action may be visualized with markers (e.g., a red circle for a click).\n\n"
        "The raw action from the execution log is provided for context:\n"
        f"- Action Type: `{log_action}`\n"
        f"- Action Detail: `{log_detail}`\n\n"
        "## Your Task\n"
        "Based on the visual evidence in the image and the provided log context, perform the following two tasks:\n"
        "1. **action_description**: In your own words, crisply describe the specific action performed (e.g., 'Clicked the final confirmation button').\n"
        "2. **ui_description**: List the key UI elements visible in this final screen state that are relevant to the action and the overall task.\n\n"
        "**Special Instruction for task completion signals:** If the `Action Detail` suggests the task has been concluded (e.g., it includes words like `completed` or `finished`), describe the action from the user's perspective, such as 'The user ended the task.' Do not state that the task *is* complete, as that is for a later evaluation step.\n\n"
        "Your output MUST be a JSON object with these two keys.\n\n"
        "### Example\n"
        "```json\n"
        "{{\n"
        '  "action_description": "The user clicked the \'Done\' button to complete the process.",\n'
        '  "ui_description": "The confirmation screen is visible, showing a success message and a \'Done\' button."\n'
        "}}\n"
        "```"
    )
    return system_prompt, user_prompt


def _get_evaluation_guidelines() -> str:
    """Returns the shared evaluation guidelines text."""
    return """## Evaluation Guidelines
1.  **Final UI State**: The "final UI state" is the conceptual state of the UI after all actions are performed. It must meet all task requirements. This state may be represented by the last screenshot, or a collection of screenshots from the middle and end of the sequence that together prove task completion. **Information Organization**: When tasks require inputting answers/information into note-taking apps, messaging apps, or similar software, the information must be organized in a logical and orderly manner. Mixed or chaotic organization (e.g., Point 1.1, Point 2.1, Point 2.2, Point 1.2) should be considered task failure, as proper information structure is essential for task completion quality.
2.  **Pre-existing Conditions**: If a task requirement was already met before the agent started (e.g., a 'Shopping' note already exists when the task is to create one), the agent does not need to repeat the action. The task is still considered successful if the final state is correct.
3.  **Trust Correct Actions**: If a sequence of actions is logically correct for the task (e.g., 'Click Save'), you can infer the action was successful and the state was achieved, even if the final screenshot shows a different screen (e.g., the agent has navigated back to the home screen).
4.  **Allow Error Correction**: The agent can make and correct mistakes. As long as the final goal is achieved, intermediate errors do not affect the outcome.
5.  **Handle Unreasonable Tasks**: If a task is inherently unreasonable or impossible to complete (e.g., requesting to find 3 reviews for a newly released product that has no reviews yet), the agent can still be considered successful if it correctly identifies the impossibility and provides appropriate feedback. For example, writing "not found", "no reviews available", or any other clear indication that the agent recognized the task's unreasonable nature is acceptable as successful task completion.
6.  **VLM Role Clarification**: The VLM is a screen UI parser — it ONLY reads and describes what is visible on the screen. It does NOT execute actions, and it does NOT generate action descriptions (those come from the agent). VLM-parsed screen descriptions may contain errors (misread numbers, omitted elements, hallucinated details); always cross-reference with raw action logs and screenshots. Reliability priority: Raw Action Log > Screenshots > VLM-parsed screen description."""


def _get_env_aware_guidelines() -> str:
    """Returns the environment-aware evaluation guidelines (v3)."""
    return _get_evaluation_guidelines() + """

7.  **Environment Robustness Evaluation**: You MUST distinguish between agent capability failures and environment-induced failures:
    - **Network Issues**: If the screenshots, action logs, or task context show signs of network connectivity problems (e.g., "network error", "connection failed", "no internet", "loading timeout", "request timed out", "504 Gateway Timeout", blank/grey screens that suggest failed content loading), this is an ENVIRONMENT failure, NOT an agent failure. The agent's reasoning and action selection may have been correct, but the environment prevented execution. In these cases, consider the task outcome as 'environment_issue'.
    - **App/Service Unavailable**: If an app crashes, shows "app not found", "service unavailable", "content blocked", "access denied", or similar error screens unrelated to the agent's actions, this is also an ENVIRONMENT failure. The agent cannot be blamed for failing to complete a task when the required app/service is unreachable. Mark such cases as 'app_not_found'.
    - **Agent Failures (Real Failures)**: Only mark a task as FAILED (decision=0) when the agent's reasoning, action selection, or execution was genuinely wrong (e.g., clicked wrong button, wrong search query, incomplete information retrieval, incorrect comparison/conclusion).
    - **Key Distinction**: Ask yourself — "Did the agent do the RIGHT thing given what it could see and know, but the environment failed it?" OR "Did the agent's reasoning/execution actually fail?" The former is an environment issue; the latter is a real failure.
    - **Evidence Sources**: Check BOTH the text descriptions (action_reasoning, action_detail, final_reason) AND the screenshots for environmental failure signals. Do NOT rely solely on the final outcome — a task that 'failed' due to network timeout is a different failure type than a task where the agent clicked the wrong button."""


def _get_base_final_decision_prompt(
    task_description: str, step_descriptions: list
 ) -> (str, str):
    """Internal helper to generate the base system prompt and formatted steps."""
    system_prompt = f"""You are an expert in evaluating mobile UI automation tasks.{_get_evaluation_guidelines()}"""

    formatted_steps = []
    for i, desc_obj in enumerate(step_descriptions):
        if isinstance(desc_obj, dict):
            step_label = f"Step {i + 1}"
            vlm_action = desc_obj.get("action_description", "N/A")
            vlm_ui = desc_obj.get("ui_description", "N/A")
            raw_action = desc_obj.get("_raw_action", {})
            raw_action_type = raw_action.get("type", "N/A")
            raw_action_detail = raw_action.get("detail", "N/A")

            formatted_steps.append(
                f"- {step_label}:\n"
                f"  - Raw Action Log: type=`{raw_action_type}`, detail=`{raw_action_detail}`\n"
                f"  - Screen UI Description (VLM-parsed): {vlm_ui}\n"
                f"  - Agent Action Description: {vlm_action}\n"
            )
        else:
            step_label = f"Step {i + 1}"
            formatted_steps.append(f"- {step_label}: {desc_obj}")
    formatted_steps_str = "\n".join(formatted_steps)

    base_user_prompt = (
        f"Task Description: '{task_description}'\n\n"
        "Here is a step-by-step breakdown of the agent's actions. Each step contains three parts:\n"
        "- **Raw Action Log**: The exact action recorded by the system (ground truth of what was executed).\n"
        "- **Screen UI Description (VLM-parsed)**: A VLM parsed the screenshot and described the UI elements visible on screen. This is ONLY a screen reading/parsing aid — the VLM does NOT execute actions and may misread numbers, omit elements, or hallucinate details.\n"
        "- **Agent Action Description**: A natural-language description of what the agent did in this step.\n\n"
        "**Reliability Priority**: Raw Action Log > Screenshots > Screen UI Description (VLM-parsed). When the VLM-parsed screen description conflicts with raw logs or screenshots, trust the raw logs and screenshots. If a critical conflict cannot be resolved (e.g., a key number differs between sources and cannot be verified), use decision = -1 to flag the conflict.\n\n"
        f"{formatted_steps_str}\n\n"
    )
    return system_prompt, base_user_prompt


def get_final_decision_prompt(
    task_description: str,
    step_descriptions: list,
    uncertainty_reason: str = "",
    last_step_thinking: str = None,
    reference_answer: str = "",
 ) -> (str, str):
    """
    Creates prompts for the detailed decision, allowing the LLM to request more screenshots.
    This phase includes step-by-step descriptions and the last 3 screenshots.
    """
    system_prompt, base_user_prompt = _get_base_final_decision_prompt(
        task_description, step_descriptions
    )

    if uncertainty_reason:
        system_prompt += f"""
 A previous, less-informed evaluation stage was 'Uncertain' for the following reason: '{uncertainty_reason}'.
 Please pay special attention to this aspect. You are now provided with more information (detailed step descriptions and the last 3 screenshots).
 """

    user_prompt = base_user_prompt

    if reference_answer:
        user_prompt += (
            f"**REFERENCE ANSWER**: The expected reference answer for this task is: '{reference_answer}'\n"
            "Use this reference answer as a benchmark to evaluate whether the agent's final output matches the expected result. "
            "If the agent's answer is semantically equivalent or numerically close to the reference answer, consider it correct.\n\n"
        )

    if last_step_thinking:
        user_prompt += (
            f"**MODEL FINAL STEP THINKING**: The following is the model's internal reasoning from its final step:\n"
            f"```\n{last_step_thinking}\n"
            f"```\n\n"
            "Please note that the above is the model's own internal reasoning/thinking process. Use this to better understand what the model intended to do in its final step. However, remember that the model's actual output/action is what matters for evaluation, not its internal reasoning.\n\n"
        )

    user_prompt += (
        "You are now provided with a composite image of the last 3 screenshots. Note that this is only a partial view of the execution. You must synthesize this visual information with the full list of text descriptions to understand the complete workflow.\n\n"
        "**CRITICAL WARNING ABOUT VLM-PARSED DESCRIPTIONS**: The 'Screen UI Description (VLM-parsed)' fields are generated by a screen-reading VLM that may misread numbers, omit UI elements, or hallucinate details. They are NOT from the task executor. When VLM-parsed descriptions conflict with raw action logs or screenshots, trust the raw logs and screenshots.\n\n"
        "**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:\n"
        "1. The text descriptions, OR\n"
        "2. The provided screenshots\n\n"
        "If critical information is missing, you should make a reasonable judgment based on the available evidence. If the evidence suggests the task was likely completed correctly, decide success; otherwise, decide failure.\n\n"
        "**CONFLICT HANDLING**: If you find a critical conflict between information sources (e.g., VLM text says the price is $29.99 but the screenshot shows $49.99, and this number is essential for task evaluation), and you cannot determine which source is correct, use decision = -1 to flag the conflict. Do NOT guess — flagging a conflict is better than making a wrong decision.\n\n"
        "**FINAL DECISION REQUIRED**: Based on all available information, respond with one of three decisions:\n"
        "- decision 1: Task completed successfully.\n"
        "- decision 0: Task failed (include 'failure_step').\n"
        "- decision -1: Critical information conflict — cannot reliably determine success or failure (include 'conflict_detail').\n\n"
        "**FAILURE STEP TRACKING**: If you determine the task failed (decision = 0), you MUST specify exactly which step number caused the failure by including a 'failure_step' field with the step number where the critical error occurred. Additionally, in your 'reason' field, you MUST include a specific explanation of why you identified this particular step as the failure point.\n\n"
        "Example (Success):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task completed successfully. The agent correctly navigated to the target app, performed the required actions, and achieved the desired outcome."\n}\n'
        "```\n\n"
        "Example (Failure):\n"
        "```json\n"
        '{\n  "decision": 0,\n  "reason": "Task failed at step 4 where wrong product was selected. Step 4 was the failure point because the agent selected an incorrect item despite correct ones being visible.",\n  "failure_step": 4\n}\n'
        "```\n\n"
        "Example (Conflict/Uncertain):\n"
        "```json\n"
        '{\n  "decision": -1,\n  "reason": "Cannot determine task outcome due to conflicting information.",\n  "conflict_detail": "VLM-parsed screen description states the price is $29.99 but the screenshot shows $49.99. The correct price is essential for evaluating whether the agent found the right product."\n}\n'
        "```"
    )
    return system_prompt, user_prompt


def get_final_decision_with_screenshots_prompt(
    task_description: str,
    step_descriptions: list,
    uncertainty_reason: str = "",
    last_step_thinking: str = None,
    reference_answer: str = "",
 ) -> (str, str):
    """
    Creates prompts for the final decision phase with supplemental screenshots.
    This is used when the LLM requests additional screenshots for clarification.
    """
    system_prompt, base_user_prompt = _get_base_final_decision_prompt(
        task_description, step_descriptions
    )
    system_prompt += f"""
 You previously requested specific screenshots for clarification because you were uncertain. The reason for uncertainty was: '{uncertainty_reason}'.
 You are now provided with a composite image showing the critical step screenshots you requested. This image is only a partial view of the execution; you must synthesize this visual information with the full list of text descriptions to understand the complete workflow.
 Based on ALL available information, you must now make a FINAL and DEFINITIVE judgment. Your decision must be success (1), failure (0), or conflict (-1). Do not request more information.
 """

    user_prompt = base_user_prompt

    if reference_answer:
        user_prompt += (
            f"**REFERENCE ANSWER**: The expected reference answer for this task is: '{reference_answer}'\n"
            "Use this reference answer as a benchmark to evaluate whether the agent's final output matches the expected result. "
            "If the agent's answer is semantically equivalent or numerically close to the reference answer, consider it correct.\n\n"
        )

    if last_step_thinking:
        user_prompt += (
            f"**MODEL FINAL STEP THINKING**: The following is the model's internal reasoning from its final step:\n"
            f"```\n{last_step_thinking}\n"
            f"```\n\n"
            "Please note that the above is the model's own internal reasoning/thinking process. Use this to better understand what the model intended to do in its final step. However, remember that the model's actual output/action is what matters for evaluation, not its internal reasoning.\n\n"
        )

    user_prompt += (
        "And here is the image with the supplemental screenshots you requested.\n\n"
        "**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:\n"
        "1. The text descriptions, OR\n"
        "2. The provided screenshots\n\n"
        "**CONFLICT HANDLING**: If you find a critical conflict between information sources (e.g., VLM text says the price is $29.99 but the screenshot shows $49.99, and this number is essential for task evaluation), and you cannot determine which source is correct, use decision = -1 to flag the conflict. Do NOT guess — flagging a conflict is better than making a wrong decision.\n\n"
        "**FINAL DECISION REQUIRED**: Based on all available information, respond with one of three decisions:\n"
        "- decision 1: Task completed successfully.\n"
        "- decision 0: Task failed (include 'failure_step').\n"
        "- decision -1: Critical information conflict — cannot reliably determine success or failure (include 'conflict_detail').\n\n"
        "**FAILURE STEP TRACKING**: If you determine the task failed (decision = 0), you MUST specify exactly which step number caused the failure by including a 'failure_step' field with the step number where the critical error occurred.\n\n"
        "Example (Success):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task completed successfully. All required actions were performed correctly."\n}\n'
        "```\n\n"
        "Example (Failure):\n"
        "```json\n"
        '{\n  "decision": 0,\n  "reason": "Task failed at step 4 where wrong product was selected.",\n  "failure_step": 4\n}\n'
        "```\n\n"
        "Example (Conflict/Uncertain):\n"
        "```json\n"
        '{\n  "decision": -1,\n  "reason": "Cannot determine task outcome due to conflicting information.",\n  "conflict_detail": "VLM-parsed screen description states the price is $29.99 but the screenshot shows $49.99. The correct price is essential for evaluating whether the agent found the right product."\n}\n'
        "```"
    )
    return system_prompt, user_prompt


def get_task_feasibility_prompt(
    task_description: str, step_descriptions: list
) -> (str, str):
    """
    Creates prompts for evaluating task feasibility/reasonableness before judging agent performance.
    """
    system_prompt = """You are an expert in evaluating mobile UI task feasibility. Your role is to determine whether a given task is inherently feasible/reasonable based on the execution context and available resources.

**Your Task**: Before evaluating agent performance, you must first assess if the task itself is reasonable and achievable given the constraints and context shown in the execution trajectory.

**Key Considerations**:
1. **Resource Availability**: Does the task require resources that are not available? (e.g., asking to share a photo when the gallery is empty, contacting a person not in contacts)
2. **Prerequisites**: Are necessary prerequisites missing? (e.g., asking to edit a specific note that doesn't exist)
3. **Logical Consistency**: Is the task internally consistent and logically achievable?
4. **Environmental Constraints**: Are there environmental factors that make the task impossible? (e.g., no internet connection when online actions are required)

**Important Notes**:
- Even if a task is deemed unreasonable/infeasible, the agent evaluation should still proceed
- If the agent correctly identifies the impossibility and provides appropriate feedback (e.g., "photo gallery is empty", "contact not found"), this should be considered successful task completion
- Only mark a task as unreasonable if there are fundamental barriers that prevent completion, not if the agent simply performed poorly

**Response Format**: You must respond with a JSON object containing:
- "feasible": boolean (true if task is reasonable and achievable, false if not)
- "reason": string explaining your assessment
- "barriers": array of specific barriers that make the task infeasible (empty if feasible)
"""

    formatted_steps = []
    for i, desc_obj in enumerate(step_descriptions):
        if isinstance(desc_obj, dict):
            step_label = f"Step {i + 1}"
            action = desc_obj.get("action_description", "N/A")
            ui = desc_obj.get("ui_description", "N/A")
            formatted_steps.append(f"- {step_label}:\n  Action: {action}\n  UI: {ui}")
        else:
            step_label = f"Step {i + 1}"
            formatted_steps.append(f"- {step_label}: {desc_obj}")
    formatted_steps_str = "\n".join(formatted_steps)

    user_prompt = (
        f"Task Description: '{task_description}'\n\n"
        "Here is the execution trajectory with step-by-step descriptions:\n"
        f"{formatted_steps_str}\n\n"
        "Based on the task description and the observed execution context, evaluate if this task is fundamentally feasible and reasonable.\n\n"
        "Examples of unreasonable tasks:\n"
        "- 'Send the first photo from gallery to John' when the photo gallery is completely empty\n"
        "- 'Edit the note titled \"Shopping List\"' when no such note exists and cannot be created\n"
        "- 'Call emergency contact' when no emergency contacts are configured\n\n"
        "Examples of reasonable tasks (even if agent fails):\n"
        "- 'Set a timer for 10 seconds' - this is always feasible if the device has a timer app\n"
        "- 'Search for restaurants nearby' - feasible if maps/search apps are available\n"
        "- 'Take a photo' - feasible if camera access is available\n\n"
        "Provide your assessment in JSON format."
    )
    return system_prompt, user_prompt


def _get_base_final_decision_prompt_v3(
    task_description: str, step_descriptions: list
) -> (str, str):
    """Internal helper to generate the v3 env-aware system prompt and formatted steps."""
    system_prompt = f"""You are an expert in evaluating mobile UI automation tasks.{_get_env_aware_guidelines()}"""

    formatted_steps = []
    for i, desc_obj in enumerate(step_descriptions):
        if isinstance(desc_obj, dict):
            step_label = f"Step {i + 1}"
            vlm_action = desc_obj.get("action_description", "N/A")
            vlm_ui = desc_obj.get("ui_description", "N/A")
            raw_action = desc_obj.get("_raw_action", {})
            raw_action_type = raw_action.get("type", "N/A")
            raw_action_detail = raw_action.get("detail", "N/A")

            formatted_steps.append(
                f"- {step_label}:\n"
                f"  - Raw Action Log: type=`{raw_action_type}`, detail=`{raw_action_detail}`\n"
                f"  - Screen UI Description (VLM-parsed): {vlm_ui}\n"
                f"  - Agent Action Description: {vlm_action}\n"
            )
        else:
            step_label = f"Step {i + 1}"
            formatted_steps.append(f"- {step_label}: {desc_obj}")
    formatted_steps_str = "\n".join(formatted_steps)

    base_user_prompt = (
        f"Task Description: '{task_description}'\n\n"
        "Here is a step-by-step breakdown of the agent's actions. Each step contains three parts:\n"
        "- **Raw Action Log**: The exact action recorded by the system (ground truth of what was executed).\n"
        "- **Screen UI Description (VLM-parsed)**: A VLM parsed the screenshot and described the UI elements visible on screen. This is ONLY a screen reading/parsing aid — the VLM does NOT execute actions and may misread numbers, omit elements, or hallucinate details.\n"
        "- **Agent Action Description**: A natural-language description of what the agent did in this step.\n\n"
        "**Reliability Priority**: Raw Action Log > Screenshots > Screen UI Description (VLM-parsed). When the VLM-parsed screen description conflicts with raw logs or screenshots, trust the raw logs and screenshots. If a critical conflict cannot be resolved (e.g., a key number differs between sources and cannot be verified), use decision = -1 to flag the conflict.\n\n"
        f"{formatted_steps_str}\n\n"
    )
    return system_prompt, base_user_prompt


def get_final_decision_prompt_v3(
    task_description: str,
    step_descriptions: list,
    uncertainty_reason: str = "",
    last_step_thinking: str = None,
    reference_answer: str = "",
) -> (str, str):
    """
    Creates v3 env-aware prompts for final decision, with network/app failure detection.
    This phase includes step-by-step descriptions and the last 3 screenshots.
    """
    system_prompt, base_user_prompt = _get_base_final_decision_prompt_v3(
        task_description, step_descriptions
    )

    if uncertainty_reason:
        system_prompt += f"""
 A previous, less-informed evaluation stage was 'Uncertain' for the following reason: '{uncertainty_reason}'.
 Please pay special attention to this aspect. You are now provided with more information (detailed step descriptions and the last 3 screenshots).
 """

    user_prompt = base_user_prompt

    if reference_answer:
        user_prompt += (
            f"**REFERENCE ANSWER**: The expected reference answer for this task is: '{reference_answer}'\n"
            "Use this reference answer as a benchmark to evaluate whether the agent's final output matches the expected result. "
            "If the agent's answer is semantically equivalent or numerically close to the reference answer, consider it correct.\n\n"
        )

    if last_step_thinking:
        user_prompt += (
            f"**MODEL FINAL STEP THINKING**: The following is the model's internal reasoning from its final step:\n"
            f"```\n{last_step_thinking}\n"
            f"```\n\n"
            "Please note that the above is the model's own internal reasoning/thinking process. Use this to better understand what the model intended to do in its final step. However, remember that the model's actual output/action is what matters for evaluation, not its internal reasoning.\n\n"
        )

    user_prompt += (
        "You are now provided with a composite image of the last 3 screenshots. Note that this is only a partial view of the execution. You must synthesize this visual information with the full list of text descriptions to understand the complete workflow.\n\n"
        "**CRITICAL WARNING ABOUT VLM-PARSED DESCRIPTIONS**: The 'Screen UI Description (VLM-parsed)' fields are generated by a screen-reading VLM that may misread numbers, omit UI elements, or hallucinate details. They are NOT from the task executor. When VLM-parsed descriptions conflict with raw action logs or screenshots, trust the raw logs and screenshots.\n\n"
        "**ENVIRONMENT FAILURE DETECTION (v3)**: You MUST actively look for evidence of environment failures in BOTH the text descriptions and screenshots. Common signals include:\n"
        "  - Network errors: 'network error', 'connection failed', 'no internet', 'loading timeout', 'request timed out', '504', '502', 'network unavailable', 'offline'\n"
        "  - App/Service errors: 'app not found', 'service unavailable', 'content blocked', 'access denied', 'crash', 'force close', 'ANR'\n"
        "  - Visual indicators in screenshots: blank/grey screens, error dialogs, loading spinners that never resolved, 'no connection' screens\n\n"
        "**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:\n"
        "1. The text descriptions, OR\n"
        "2. The provided screenshots\n\n"
        "If critical information is missing, you should make a reasonable judgment based on the available evidence. If the evidence suggests the task was likely completed correctly, decide success; otherwise, decide failure.\n\n"
        "**CONFLICT HANDLING**: If you find a critical conflict between information sources (e.g., VLM text says the price is $29.99 but the screenshot shows $49.99, and this number is essential for task evaluation), and you cannot determine which source is correct, use decision = -1 to flag the conflict. Do NOT guess — flagging a conflict is better than making a wrong decision.\n\n"
        "**FINAL DECISION REQUIRED**: Based on all available information, respond with one of four decisions:\n"
        "- decision 1: Task completed successfully.\n"
        "- decision 0: Task failed — agent's reasoning or execution was wrong (include 'failure_step').\n"
        "- decision -1: Critical information conflict — cannot reliably determine success or failure (include 'conflict_detail').\n"
        "- decision -2: Task failed due to environment issues (network/app failure), NOT agent capability failure. The agent may have reasoned correctly but the environment prevented completion (include 'failure_step' and set 'failure_type'='environment_issue').\n\n"
        "**FAILURE STEP TRACKING**: If you determine the task failed (decision = 0 or -2), you MUST specify exactly which step number caused the failure by including a 'failure_step' field with the step number where the critical error occurred. Additionally, in your 'reason' field, you MUST include a specific explanation of why you identified this particular step as the failure point. For environment issues (-2), clearly state what environmental factor caused the failure.\n\n"
        "Example (Success):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task completed successfully. The agent correctly navigated to the target app, performed the required actions, and achieved the desired outcome."\n}\n'
        "```\n\n"
        "Example (Agent Failure):\n"
        "```json\n"
        '{\n  "decision": 0,\n  "reason": "Task failed at step 4 where wrong product was selected. Step 4 was the failure point because the agent selected an incorrect item despite correct ones being visible.",\n  "failure_step": 4\n}\n'
        "```\n\n"
        "Example (Environment Failure - Network):\n"
        "```json\n"
        "{\n  \"decision\": -2,\n  \"reason\": \"Task failed at step 3 due to network timeout. The agent correctly searched for the product and was about to click on it, but the page failed to load with a 504 Gateway Timeout error. The agent cannot be blamed for this failure.\",\n  \"failure_step\": 3,\n  \"failure_type\": \"environment_issue\"\n}\n"
        "```\n\n"
        "Example (Conflict/Uncertain):\n"
        "```json\n"
        '{\n  "decision": -1,\n  "reason": "Cannot determine task outcome due to conflicting information.",\n  "conflict_detail": "VLM-parsed screen description states the price is $29.99 but the screenshot shows $49.99. The correct price is essential for evaluating whether the agent found the right product."\n}\n'
        "```"
    )
    return system_prompt, user_prompt


def get_final_decision_with_screenshots_prompt_v3(
    task_description: str,
    step_descriptions: list,
    uncertainty_reason: str = "",
    last_step_thinking: str = None,
    reference_answer: str = "",
) -> (str, str):
    """
    Creates v3 env-aware prompts for final decision with supplemental screenshots.
    This is used when the LLM requests additional screenshots for clarification.
    """
    system_prompt, base_user_prompt = _get_base_final_decision_prompt_v3(
        task_description, step_descriptions
    )
    system_prompt += f"""
 You previously requested specific screenshots for clarification because you were uncertain. The reason for uncertainty was: '{uncertainty_reason}'.
 You are now provided with a composite image showing the critical step screenshots you requested. This image is only a partial view of the execution; you must synthesize this visual information with the full list of text descriptions to understand the complete workflow.
 Based on ALL available information, you must now make a FINAL and DEFINITIVE judgment. Your decision must be success (1), failure (0), conflict (-1), or environment_failure (-2). Do not request more information.
 """

    user_prompt = base_user_prompt

    if reference_answer:
        user_prompt += (
            f"**REFERENCE ANSWER**: The expected reference answer for this task is: '{reference_answer}'\n"
            "Use this reference answer as a benchmark to evaluate whether the agent's final output matches the expected result. "
            "If the agent's answer is semantically equivalent or numerically close to the reference answer, consider it correct.\n\n"
        )

    if last_step_thinking:
        user_prompt += (
            f"**MODEL FINAL STEP THINKING**: The following is the model's internal reasoning from its final step:\n"
            f"```\n{last_step_thinking}\n"
            f"```\n\n"
            "Please note that the above is the model's own internal reasoning/thinking process. Use this to better understand what the model intended to do in its final step. However, remember that the model's actual output/action is what matters for evaluation, not its internal reasoning.\n\n"
        )

    user_prompt += (
        "And here is the image with the supplemental screenshots you requested.\n\n"
        "**ENVIRONMENT FAILURE DETECTION (v3)**: You MUST actively look for evidence of environment failures in BOTH the text descriptions and screenshots. Common signals include:\n"
        "  - Network errors: 'network error', 'connection failed', 'no internet', 'loading timeout', 'request timed out', '504', '502', 'network unavailable', 'offline'\n"
        "  - App/Service errors: 'app not found', 'service unavailable', 'content blocked', 'access denied', 'crash', 'force close', 'ANR'\n"
        "  - Visual indicators in screenshots: blank/grey screens, error dialogs, loading spinners that never resolved, 'no connection' screens\n\n"
        "**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:\n"
        "1. The text descriptions, OR\n"
        "2. The provided screenshots\n\n"
        "**CONFLICT HANDLING**: If you find a critical conflict between information sources (e.g., VLM text says the price is $29.99 but the screenshot shows $49.99, and this number is essential for task evaluation), and you cannot determine which source is correct, use decision = -1 to flag the conflict. Do NOT guess — flagging a conflict is better than making a wrong decision.\n\n"
        "**FINAL DECISION REQUIRED**: Based on all available information, respond with one of four decisions:\n"
        "- decision 1: Task completed successfully.\n"
        "- decision 0: Task failed — agent's reasoning or execution was wrong (include 'failure_step').\n"
        "- decision -1: Critical information conflict — cannot reliably determine success or failure (include 'conflict_detail').\n"
        "- decision -2: Task failed due to environment issues (network/app failure), NOT agent capability failure (include 'failure_step' and 'failure_type'='environment_issue').\n\n"
        "**FAILURE STEP TRACKING**: If you determine the task failed (decision = 0 or -2), you MUST specify exactly which step number caused the failure by including a 'failure_step' field with the step number where the critical error occurred.\n\n"
        "Example (Success):\n"
        "```json\n"
        '{\n  "decision": 1,\n  "reason": "Task completed successfully. All required actions were performed correctly."\n}\n'
        "```\n\n"
        "Example (Agent Failure):\n"
        "```json\n"
        '{\n  "decision": 0,\n  "reason": "Task failed at step 4 where wrong product was selected.",\n  "failure_step": 4\n}\n'
        "```\n\n"
        "Example (Environment Failure):\n"
        "```json\n"
        '{\n  "decision": -2,\n  "reason": "Task failed at step 3 due to network timeout. The agent correctly searched for the product.",\n  "failure_step": 3,\n  "failure_type": "environment_issue"\n}\n'
        "```\n\n"
        "Example (Conflict/Uncertain):\n"
        "```json\n"
        '{\n  "decision": -1,\n  "reason": "Cannot determine task outcome due to conflicting information.",\n  "conflict_detail": "VLM-parsed screen description states the price is $29.99 but the screenshot shows $49.99."\n}\n'
        "```"
    )
    return system_prompt, user_prompt


def get_pre_evaluation_prompt(
    task_description: str, raw_action_logs: list, total_steps: int
) -> (str, str):
    """
    Creates system and user prompts for the pre-evaluation phase.
    """
    system_prompt = f"""You are an expert in evaluating mobile UI automation tasks. Your goal is to determine if a task has DEFINITELY succeeded based on VERY limited information. You must be extremely confident to make a "Success" decision.

{_get_evaluation_guidelines()}

You will be given:
1. The task description.
2. The raw action logs (without semantic descriptions).
3. A single image combining the last 3 screenshots out of a total of {total_steps} screenshots.

**Crucial Instructions:**
- The information provided is INCOMPLETE. You are only seeing the final UI states and raw, low-level actions.
- You must be EXTREMELY conservative. Only conclude "Success" if the provided evidence is undeniable and accounts for ALL conditions in the task description with absolute certainty.
- If there is ANY ambiguity or any task condition that cannot be verified from the final screenshots (e.g., a filter that was applied in an earlier step), you MUST respond with "Uncertain" and provide a reason. You cannot decide "Failure" at this stage.

**MANDATORY VERIFICATION**: Before making any decision, you MUST verify that ALL key information required by the task description is present in either:
1. The raw action logs, OR
2. The provided screenshots

If ANY critical information, parameters, values, or UI elements mentioned in the task description are NOT clearly visible in the provided screenshots and NOT evident from the raw action logs, you MUST respond with "Uncertain". Do not guess or infer missing information. All required information must be explicitly present and verifiable.

Example Scenarios for "Uncertain":
- Task: "In Amazon, search for 'laptop', filter by '4 stars & up', and add the first item to the cart."
- Provided Info: The final screenshot shows an item in the shopping cart.
- Correct Response: "Uncertain". The final screenshot proves an item was added to the cart, but it's impossible to verify if the '4 stars & up' filter was correctly applied. This requires more information.

- Task: "In Amazon, search for three products A, B, and C in sequence, remember their prices and star ratings, then write down the information you just found in a note-taking app."
- Provided Info: The final screenshots show a note-taking app with prices and ratings for products A, B, and C.
- Correct Response: "Uncertain". While the final screenshots show that information was written in the note-taking app, it's impossible to verify if the recorded prices and ratings actually match the real search results from earlier steps, since the search result screenshots are not provided.

Respond with a JSON object containing "reason" and "decision" ("Success" or "Uncertain").
"""
    raw_actions_str = "\n".join([f"- {log}" for log in raw_action_logs])
    user_prompt = f"""Task Description:
{task_description}

Total Steps in Full Trajectory: {total_steps}
Raw Action Sequence:
{raw_actions_str}

Please evaluate the task outcome based on the provided image showing only the final UI states and the raw action logs.
"""

    return system_prompt, user_prompt
