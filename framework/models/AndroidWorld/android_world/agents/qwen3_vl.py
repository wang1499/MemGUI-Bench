# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen3-VL: Vision-Language Model Agent for Android."""

import base64
import cv2
import io
import json
import re
import time
from typing import Any

from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.agents.retry_utils import RetryableAPIClient
from android_world.env import interface
from android_world.env import json_action


# ANSI颜色代码
class Colors:
    """终端颜色工具类"""

    RED = "\033[91m"  # 错误
    GREEN = "\033[92m"  # 成功
    YELLOW = "\033[93m"  # 警告
    BLUE = "\033[94m"  # 信息
    MAGENTA = "\033[95m"  # 重要信息
    CYAN = "\033[96m"  # 步骤/调试
    BOLD = "\033[1m"  # 粗体
    RESET = "\033[0m"  # 重置

    @staticmethod
    def error(text: str) -> str:
        """红色错误信息"""
        return f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.RESET}"

    @staticmethod
    def success(text: str) -> str:
        """绿色成功信息"""
        return f"{Colors.GREEN}{Colors.BOLD}✓ {text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        """黄色警告信息"""
        return f"{Colors.YELLOW}{Colors.BOLD}⚠ {text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        """蓝色信息"""
        return f"{Colors.BLUE}{text}{Colors.RESET}"

    @staticmethod
    def step(text: str) -> str:
        """青色步骤信息"""
        return f"{Colors.CYAN}{Colors.BOLD}🔹 {text}{Colors.RESET}"

    @staticmethod
    def important(text: str) -> str:
        """紫色重要信息"""
        return f"{Colors.MAGENTA}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        """标题"""
        return f"{Colors.CYAN}{Colors.BOLD}{'=' * 60}\n{text}\n{'=' * 60}{Colors.RESET}"

from android_world.agents import qwen_propmt
SYSTEM_PROMPT = qwen_propmt.SYSTEM_PROMPT_BASE




class Qwen3VL(base_agent.EnvironmentInteractingAgent):
    """Qwen3-VL agent for Android using vision-language model."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        config: dict[str, Any],
        name: str = "Qwen3VL",
    ):
        """Initializes a Qwen3VL agent.

        Args:
          env: The environment.
          config: Configuration dictionary containing 'QWEN_BASE_URL', 'QWEN_API_KEY',
                  and 'QWEN_MODEL'.
          name: The agent name.
        """
        super().__init__(env, name)

        if not config.get("QWEN_BASE_URL"):
            raise ValueError("QWEN_BASE_URL is required in config")
        if not config.get("QWEN_API_KEY"):
            raise ValueError("QWEN_API_KEY is required in config")
        if not config.get("QWEN_MODEL"):
            raise ValueError("QWEN_MODEL is required in config")

        base_url = config["QWEN_BASE_URL"]
        api_key = config["QWEN_API_KEY"]
        model_name = config["QWEN_MODEL"]

        # Wrap the client with retry capability for rate limit handling
        # max_retries=None means infinite retries until success
        raw_client = OpenAI(base_url=base_url, api_key=api_key)
        self.client = RetryableAPIClient(
            raw_client,
            max_retries=None,  # Infinite retries until success
            base_delay=2.0,
            max_delay=120.0,
            verbose=True,
        )
        self.model_name = model_name
        self.history = []
        self.qwen3_vl_operations = []
        self.qwen3_vl_step_data = []

        # Enhanced logging for model interactions (similar to UITARS)
        self.detailed_model_logs = []  # Store complete input/output for each model call
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def reset(self, go_home_on_reset: bool = False):
        """Resets the agent."""
        super().reset(go_home_on_reset)
        self.history = []
        self.qwen3_vl_operations = []
        self.qwen3_vl_step_data = []

        # Reset enhanced logging
        self.detailed_model_logs = []
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def _extract_conclusion(self, output: str) -> str:
        """Extract conclusion text from model output.

        If <conclusion> tags are missing, falls back to extracting text from
        <thinking> tags.

        Args:
          output: Raw output from the model.

        Returns:
          Text within <conclusion> tags, or <thinking> tags if conclusion is missing,
          or empty string if neither is found.
        """
        conclusion_match = re.search(
            r"<conclusion>\s*(.*?)\s*</conclusion>", output, re.DOTALL | re.IGNORECASE
        )
        if conclusion_match:
            return conclusion_match.group(1).strip()

        # Fallback to thinking if conclusion is missing
        thinking_match = re.search(
            r"<thinking>\s*(.*?)\s*</thinking>", output, re.DOTALL | re.IGNORECASE
        )
        if thinking_match:
            return thinking_match.group(1).strip()

        return ""

    def _resize_screenshot(self, screenshot_base64: str) -> str:
        """Return the original screenshot as it is.

        Since no resizing is needed, this function simply returns the input.

        Args:
          screenshot_base64: Base64 encoded screenshot

        Returns:
          The original screenshot as base64 string
        """
        return screenshot_base64

    def _sanitize_messages_for_logging(self, messages):
        """
        Keep original messages for logging without any sanitization.
        This preserves the complete base64 image data for debugging purposes.

        Args:
          messages: List of message dictionaries

        Returns:
          The original messages without any modification
        """
        # Return the original messages without any modification
        return messages

    def get_enhanced_log_data(self):
        """
        Get enhanced logging data including detailed model interactions.
        This method should be called when saving execution results.

        Returns:
          Dictionary containing detailed model logs and statistics
        """
        return {
            "detailed_model_logs": self.detailed_model_logs,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_model_calls": len(self.detailed_model_logs),
        }

    def _parse_qwen3_vl_output(
        self, output: str, screen_width: int, screen_height: int
    ) -> json_action.JSONAction | None:
        """Parse Qwen3-VL model output and convert to JSONAction.

        Args:
          output: Raw output from the model
          screen_width: Screen width in pixels
          screen_height: Screen height in pixels

        Returns:
          JSONAction object or None if parsing fails
        """
        # Extract tool call from output
        action_match = re.search(
            r"<tool_call>\s*(.*)\s*</tool_call>", output, re.DOTALL
        )
        if not action_match:
            print(Colors.error(f"No tool call found in output"))
            print(Colors.warning(f"Model output: {output[:200]}..."))
            return None

        try:
            action_dict = json.loads(action_match.group(1).strip())
            # Handle both formats:
            # 1. Standard: {"name": "mobile_use", "arguments": {"action": "...", ...}}
            # 2. Simplified: {"name": "mobile_use", "action": "...", ...}
            if "arguments" in action_dict:
                action_args = action_dict.get("arguments", {})
            else:
                # If no "arguments" key, treat the whole dict (except "name") as arguments
                action_args = {k: v for k, v in action_dict.items() if k != "name"}
        except json.JSONDecodeError as e:
            print(Colors.error(f"Failed to parse JSON: {e}"))
            print(Colors.warning(f"Raw content: {action_match.group(1).strip()[:200]}"))
            return None

        action_type = action_args.get("action")

        # Convert GUI-Owl actions to JSONAction format
        if action_type == "click":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x, y = int(coords[0]), int(coords[1])
                x = int(x * screen_width / 1000)
                y = int(y * screen_height / 1000)
                return json_action.JSONAction(action_type="click", x=x, y=y)

        elif action_type == "long_press":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x, y = int(coords[0]), int(coords[1])
                x = int(x * screen_width / 1000)
                y = int(y * screen_height / 1000)
                # Note: long_press duration is hardcoded in adb_utils, so we don't pass it
                return json_action.JSONAction(action_type="long_press", x=x, y=y)

        elif action_type == "swipe":
            coords = action_args.get("coordinate", [None, None])
            coords2 = action_args.get("coordinate2", [None, None])

            # Check if it's a swipe with start and end coordinates (which maps to drag in AndroidWorld)
            if (
                len(coords) >= 2
                and len(coords2) >= 2
                and coords[0] is not None
                and coords2[0] is not None
            ):
                x1 = int(coords[0] * screen_width / 1000)
                y1 = int(coords[1] * screen_height / 1000)
                x2 = int(coords2[0] * screen_width / 1000)
                y2 = int(coords2[1] * screen_height / 1000)

                return json_action.JSONAction(
                    action_type="drag", coordinate1=(x1, y1), coordinate2=(x2, y2)
                )

            direction = action_args.get("direction")

            # Map Qwen3-VL directions to AndroidWorld conventions
            direction_map = {
                "up": "up",
                "down": "down",
                "left": "left",
                "right": "right",
            }

            if direction in direction_map:
                mapped_direction = direction_map[direction]

                if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                    x, y = int(coords[0]), int(coords[1])
                    x = int(x * screen_width / 1000)
                    y = int(y * screen_height / 1000)
                    return json_action.JSONAction(
                        action_type="swipe",
                        x=x,
                        y=y,
                        direction=mapped_direction,
                    )
                else:
                    return json_action.JSONAction(
                        action_type="swipe",
                        direction=mapped_direction,
                    )

        elif action_type == "drag":
            coords1 = action_args.get("coordinate", [None, None])
            coords2 = action_args.get("coordinate2", [None, None])

            if (
                len(coords1) >= 2
                and len(coords2) >= 2
                and coords1[0] is not None
                and coords2[0] is not None
            ):
                x1 = int(coords1[0] * screen_width / 1000)
                y1 = int(coords1[1] * screen_height / 1000)
                x2 = int(coords2[0] * screen_width / 1000)
                y2 = int(coords2[1] * screen_height / 1000)

                return json_action.JSONAction(
                    action_type="drag", coordinate1=(x1, y1), coordinate2=(x2, y2)
                )

        elif action_type == "type":
            text = action_args.get("text", "")
            if text:
                return json_action.JSONAction(action_type="input_text", text=text)

        elif action_type == "answer":
            text = action_args.get("text", "")
            return json_action.JSONAction(action_type="answer", text=text)

        elif action_type == "system_button":
            button = action_args.get("button", "")
            if button == "Back":
                return json_action.JSONAction(action_type="navigate_back")
            elif button == "Home":
                return json_action.JSONAction(action_type="navigate_home")
            elif button == "Menu":
                return json_action.JSONAction(action_type="navigate_menu")
            elif button == "Enter":
                return json_action.JSONAction(action_type="keyboard_enter")

        elif action_type == "open":
            app_name = action_args.get("text", "").lower()
            if app_name:
                return json_action.JSONAction(action_type="open_app", app_name=app_name)

        elif action_type == "wait":
            # Note: wait action in actuation.py sleeps for 1 second, ignoring time parameter
            return json_action.JSONAction(action_type="wait")

        elif action_type == "terminate":
            status = action_args.get("status", "")
            if status == "success":
                return json_action.JSONAction(
                    action_type="status", goal_status="complete"
                )
            else:
                return json_action.JSONAction(
                    action_type="status", goal_status="infeasible"
                )

        print(Colors.error(f"Unknown or malformed action: {action_type}"))
        return None

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Performs a step of the agent on the environment.

        Args:
          goal: The goal/task description.

        Returns:
          AgentInteractionResult containing done status and step data.
        """
        step_data = {
            "before_screenshot": None,
            "action_output": None,
            "raw_response": None,
        }

        step_num = len(self.history)
        print(Colors.header(f"Step {step_num + 1}"))

        # Get current state
        state = self.get_post_transition_state()
        step_data["before_screenshot"] = state.pixels.copy()

        # Convert screenshot to base64
        _, buffer = cv2.imencode(".png", state.pixels)
        screenshot_base64 = base64.b64encode(buffer).decode("utf-8")

        # Resize screenshot
        screenshot_resized = self._resize_screenshot(screenshot_base64)

        # Build action history string (keep for compatibility)
        action_history_str = []
        for i, hist_item in enumerate(self.history):
            action_summary = hist_item.get("action_summary", "unknown")
            action_history_str.append(f"Step{i + 1}: {action_summary}")
        action_history_str = (
            ", ".join(action_history_str) if action_history_str else "None"
        )

        # Build action_history_desc from conclusions
        action_history_desc = []
        for hist_item in self.history:
            conclusion = hist_item.get("conclusion", "")
            if conclusion:
                action_history_desc.append(conclusion)

        # Format action_history_desc for prompt
        if action_history_desc:
            action_history_desc_str = "\n".join(
                [f"Step {i + 1}: {desc}" for i, desc in enumerate(action_history_desc)]
            )
        else:
            action_history_desc_str = ""

        # Create user prompt using action_history_desc
        user_prompt = (
            f"The user query: {goal}\n"
            f"Task progress (You have done the following operation on the current device): "
            f"{action_history_desc_str}<image>"
            ". Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, and insert them before the <tool_call></tool_call> XML tags. After answering, summarize your action in <conclusion></conclusion> tags, and insert them after the <tool_call></tool_call> XML tags."
        )

        step_data["user_prompt"] = user_prompt
        step_data["system_prompt"] = SYSTEM_PROMPT
        print(Colors.step("User prompt:"))
        print(Colors.info(f"{user_prompt}\n"))

        # Prepare messages for API call
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image;base64,{screenshot_resized}"},
                    },
                ],
            },
        ]

        # Record the API call details for enhanced logging
        api_call_start_time = time.time()
        model_log_entry = {
            "step": step_num + 1,
            "retry_count": 1,  # Qwen3VL doesn't retry by default, but keep for consistency
            "timestamp": api_call_start_time,
            "input_messages": self._sanitize_messages_for_logging(api_messages),
            "model": self.model_name,
            "raw_response": None,
            "parsed_action": None,
            "success": False,
            "error": None,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_call_duration": 0.0,
        }

        # Call Qwen3-VL model with retry mechanism
        try:
            response = self.client.create_chat_completion(
                model=self.model_name,
                messages=api_messages,
            )

            api_call_end_time = time.time()
            model_log_entry["api_call_duration"] = (
                api_call_end_time - api_call_start_time
            )

            response_str = response.choices[0].message.content or ""
            model_log_entry["raw_response"] = response_str

            # Extract reasoning_content for Thinking models (e.g., Qwen3-VL Thinking)
            reasoning_content = None
            message = response.choices[0].message
            if hasattr(message, "reasoning_content") and message.reasoning_content:
                reasoning_content = message.reasoning_content
            elif hasattr(message, "reasoning") and message.reasoning:
                reasoning_content = message.reasoning
            elif hasattr(message, "model_extra") and message.model_extra:
                reasoning_content = message.model_extra.get(
                    "reasoning_content"
                ) or message.model_extra.get("reasoning")

            if reasoning_content:
                model_log_entry["reasoning_content"] = reasoning_content
                print(
                    f"[Thinking Model] Reasoning content extracted ({len(reasoning_content)} chars)"
                )

            # Extract token usage information - save complete usage object
            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                model_log_entry["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                model_log_entry["completion_tokens"] = getattr(
                    usage, "completion_tokens", 0
                )
                model_log_entry["total_tokens"] = getattr(usage, "total_tokens", 0)

                # Save complete usage object to prevent missing any fields
                try:
                    if hasattr(usage, "model_dump"):
                        model_log_entry["usage_raw"] = usage.model_dump()
                    elif hasattr(usage, "dict"):
                        model_log_entry["usage_raw"] = usage.dict()
                    elif hasattr(usage, "__dict__"):
                        model_log_entry["usage_raw"] = dict(usage.__dict__)
                    else:
                        model_log_entry["usage_raw"] = {
                            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
                            "completion_tokens": getattr(usage, "completion_tokens", 0),
                            "total_tokens": getattr(usage, "total_tokens", 0),
                        }
                except Exception:
                    model_log_entry["usage_raw"] = str(usage)

                # Update total counters
                self.total_prompt_tokens += model_log_entry["prompt_tokens"]
                self.total_completion_tokens += model_log_entry["completion_tokens"]

                # Also save usage_raw to step_data for log.json
                step_data["usage_raw"] = model_log_entry.get("usage_raw")

        except Exception as e:
            print(Colors.error(f"Error calling Qwen3-VL model (after retries): {e}"))
            model_log_entry["error"] = str(e)
            model_log_entry["api_call_duration"] = time.time() - api_call_start_time
            self.detailed_model_logs.append(model_log_entry)
            step_data["raw_response"] = str(e)
            # Ensure parsed_action is set to avoid KeyError in downstream processing
            step_data["parsed_action"] = {"action_type": "error", "error": str(e)}
            step_data["action_summary"] = f"Model call failed: {str(e)[:100]}"
            step_data["conclusion"] = ""
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.important("Qwen3-VL output:"))
        print(Colors.info(f"{response_str}\n"))
        step_data["action_output"] = response_str
        step_data["raw_response"] = response_str

        # Store reasoning_content for Thinking models
        if reasoning_content:
            step_data["reasoning_content"] = reasoning_content
            print(f"[Thinking Model] Reasoning preview: {reasoning_content[:200]}...")

        # Extract conclusion from response
        conclusion = self._extract_conclusion(response_str)
        step_data["conclusion"] = conclusion

        # Parse output to action
        screen_size = self.env.device_screen_size
        action = self._parse_qwen3_vl_output(
            response_str, screen_size[0], screen_size[1]
        )

        if action is None:
            print(Colors.error("Failed to parse action from Qwen3-VL output"))
            step_data["action_summary"] = "Failed to parse action"
            step_data["parsed_action"] = {}
            action_history_desc.append("Failed to parse action")
            step_data["action_history_desc"] = action_history_desc.copy()

            # Log the failure
            model_log_entry["parsed_action"] = None
            model_log_entry["error"] = "Failed to parse action from output"
            self.detailed_model_logs.append(model_log_entry)

            self.history.append(step_data)
            self._update_qwen3_vl_data(goal, step_num + 1, step_data, {}, conclusion)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.success(f"Parsed action: {action.action_type}"))
        if hasattr(action, "x") and action.x is not None:
            print(Colors.info(f"  Coordinates: ({action.x}, {action.y})"))

        # Convert action to dict for storage
        parsed_action_dict = {
            "action_type": action.action_type,
        }
        if hasattr(action, "x") and action.x is not None:
            parsed_action_dict["x"] = action.x
        if hasattr(action, "y") and action.y is not None:
            parsed_action_dict["y"] = action.y
        if hasattr(action, "text") and action.text is not None:
            parsed_action_dict["text"] = action.text
        if hasattr(action, "direction") and action.direction is not None:
            parsed_action_dict["direction"] = action.direction
        if hasattr(action, "coordinate1") and action.coordinate1 is not None:
            parsed_action_dict["coordinate1"] = action.coordinate1
        if hasattr(action, "coordinate2") and action.coordinate2 is not None:
            parsed_action_dict["coordinate2"] = action.coordinate2
        if hasattr(action, "goal_status") and action.goal_status is not None:
            parsed_action_dict["goal_status"] = action.goal_status

        step_data["parsed_action"] = parsed_action_dict

        # Log successful parsing
        model_log_entry["parsed_action"] = parsed_action_dict
        model_log_entry["success"] = True
        self.detailed_model_logs.append(model_log_entry)

        # Check if task is complete
        if action.action_type == "status":
            step_data["action_summary"] = (
                f"Task completed with status: {action.goal_status}"
            )
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(True, step_data)
        elif action.action_type == "answer":
            step_data["action_summary"] = f"Answer: {action.text}"
            # Execute answer action to save the answer to interaction_cache
            try:
                actual_action_coordinates = self.env.execute_action(action)
                step_data["actual_action_coordinates"] = actual_action_coordinates
                print(Colors.success(f"Executed answer action"))
                print(Colors.info(f"  Answer text: {action.text}"))
            except Exception as e:
                print(Colors.error(f"Error executing answer action: {e}"))
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            should_end_task = "answer" in goal.lower()
            return base_agent.AgentInteractionResult(should_end_task, step_data)

        # Execute action
        try:
            actual_action_coordinates = self.env.execute_action(action)
            step_data["actual_action_coordinates"] = actual_action_coordinates
            step_data["action_summary"] = f"{action.action_type}"
            print(Colors.success(f"Executed action: {action.action_type}"))
            time.sleep(2.0)
        except Exception as e:
            print(Colors.error(f"Error executing action: {e}"))
            step_data["action_summary"] = (
                f"Error executing {action.action_type}: {str(e)}"
            )
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(False, step_data)

        # Add conclusion to action_history_desc
        action_history_desc.append(
            conclusion if conclusion else step_data["action_summary"]
        )
        step_data["action_history_desc"] = action_history_desc.copy()

        self.history.append(step_data)
        self._update_qwen3_vl_data(
            goal, step_num + 1, step_data, parsed_action_dict, conclusion
        )

        # Add Qwen3 VL specific data to step_data for checkpointer
        step_data["qwen3_vl_operations"] = self.qwen3_vl_operations
        step_data["qwen3_vl_step_data"] = self.qwen3_vl_step_data

        return base_agent.AgentInteractionResult(False, step_data)

    def _update_qwen3_vl_data(
        self,
        goal: str,
        step_id: int,
        step_data: dict,
        parsed_action: dict,
        conclusion: str,
    ):
        """Update Qwen3 VL specific data structures.

        Args:
          goal: The goal/task description.
          step_id: Current step number.
          step_data: Step data dictionary.
          parsed_action: Parsed action dictionary.
          conclusion: Extracted conclusion text.
        """
        # Update operations list
        self.qwen3_vl_operations = {
            "instruction": goal,
            "episode_id": f"episode_0",
            "steps": [],
        }

        for i, hist_item in enumerate(self.history):
            step_info = {
                "step_id": i + 1,
                "image_path": f"./screenshots/step_{i:02d}.png",
                "action": hist_item.get("parsed_action", {}),
                "conclusion": hist_item.get("conclusion", ""),
            }
            self.qwen3_vl_operations["steps"].append(step_info)

        # Update step data list
        self.qwen3_vl_step_data = []
        for i, hist_item in enumerate(self.history):
            step_entry = {
                "step_id": i + 1,
                "screenshot_path": f"./screenshots/step_{i:02d}.png",
                "action_prompt": hist_item.get("user_prompt", ""),
                "system_prompt": hist_item.get("system_prompt", ""),
                "action_output": hist_item.get("raw_response", ""),
                "parsed_action": hist_item.get("parsed_action", {}),
                "conclusion": hist_item.get("conclusion", ""),
                "action_history_desc": hist_item.get("action_history_desc", []),
            }
            self.qwen3_vl_step_data.append(step_entry)
