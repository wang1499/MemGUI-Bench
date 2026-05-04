"""Qwen3-VL v1 agent with mobile, todo and memory tool support."""

import base64
import copy
import cv2
import json
import re
import time
from typing import Any

from android_world.agents import base_agent
from android_world.agents import qwen3_vl
from android_world.agents import qwen_propmt
from android_world.env import interface
from android_world.env import json_action


Colors = qwen3_vl.Colors

SYSTEM_PROMPT_V1 = qwen_propmt.SYSTEM_PROMPT_V1

class Qwen3VLV1(qwen3_vl.Qwen3VL):
    """Qwen3-VL v1 agent with todo and memory tools."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        config: dict[str, Any],
        name: str = "Qwen3VLV1",
    ):
        super().__init__(env, config, name)
        self.todo_state: list[dict[str, Any]] = []
        self.memory_state: dict[str, dict[str, Any]] = {}
        self.tool_call_stats = {
            "mobile_use": 0,
            "write_todos": 0,
            "write_memory": 0,
        }

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.todo_state = []
        self.memory_state = {}
        self.tool_call_stats = {
            "mobile_use": 0,
            "write_todos": 0,
            "write_memory": 0,
        }

    def _format_todo_state(self) -> str:
        if not self.todo_state:
            return "None"
        return json.dumps(self.todo_state, ensure_ascii=False)

    def _format_memory_state(self) -> str:
        if not self.memory_state:
            return "None"
        ordered_items = [self.memory_state[k] for k in sorted(self.memory_state.keys())]
        return json.dumps(ordered_items, ensure_ascii=False)

    def _extract_tool_call_blocks(self, output: str) -> list[str]:
        return re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.DOTALL)

    def _normalize_tool_call_dict(self, tool_call_str: str) -> dict[str, Any]:
        tool_call = json.loads(tool_call_str.strip())
        if "name" not in tool_call:
            raise ValueError("Tool call missing 'name'")
        if "arguments" not in tool_call:
            tool_call["arguments"] = {
                k: v for k, v in tool_call.items() if k not in {"name", "arguments"}
            }
        if not isinstance(tool_call["arguments"], dict):
            raise ValueError("Tool call 'arguments' must be an object")
        return tool_call

    def _snapshot_memory_state(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.memory_state[k]) for k in sorted(self.memory_state)]

    def _action_to_dict(self, action: json_action.JSONAction) -> dict[str, Any]:
        parsed_action_dict = {"action_type": action.action_type}
        if getattr(action, "x", None) is not None:
            parsed_action_dict["x"] = action.x
        if getattr(action, "y", None) is not None:
            parsed_action_dict["y"] = action.y
        if getattr(action, "text", None) is not None:
            parsed_action_dict["text"] = action.text
        if getattr(action, "direction", None) is not None:
            parsed_action_dict["direction"] = action.direction
        if getattr(action, "coordinate1", None) is not None:
            parsed_action_dict["coordinate1"] = action.coordinate1
        if getattr(action, "coordinate2", None) is not None:
            parsed_action_dict["coordinate2"] = action.coordinate2
        if getattr(action, "goal_status", None) is not None:
            parsed_action_dict["goal_status"] = action.goal_status
        if getattr(action, "app_name", None) is not None:
            parsed_action_dict["app_name"] = action.app_name
        return parsed_action_dict

    def _parse_mobile_use_arguments(
        self, action_args: dict[str, Any], screen_width: int, screen_height: int
    ) -> json_action.JSONAction | None:
        action_type = action_args.get("action")

        if action_type == "click":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x = int(coords[0] * screen_width / 1000)
                y = int(coords[1] * screen_height / 1000)
                return json_action.JSONAction(action_type="click", x=x, y=y)

        elif action_type == "long_press":
            coords = action_args.get("coordinate", [None, None])
            if len(coords) >= 2 and coords[0] is not None and coords[1] is not None:
                x = int(coords[0] * screen_width / 1000)
                y = int(coords[1] * screen_height / 1000)
                return json_action.JSONAction(action_type="long_press", x=x, y=y)

        elif action_type == "swipe":
            coords = action_args.get("coordinate", [None, None])
            coords2 = action_args.get("coordinate2", [None, None])
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

        elif action_type == "type":
            text = action_args.get("text", "")
            if text:
                return json_action.JSONAction(action_type="input_text", text=text)

        elif action_type == "answer":
            return json_action.JSONAction(
                action_type="answer", text=action_args.get("text", "")
            )

        elif action_type == "system_button":
            button = action_args.get("button", "")
            button_mapping = {
                "Back": "navigate_back",
                "Home": "navigate_home",
                "Menu": "navigate_menu",
                "Enter": "keyboard_enter",
            }
            mapped_type = button_mapping.get(button)
            if mapped_type:
                return json_action.JSONAction(action_type=mapped_type)

        elif action_type == "wait":
            return json_action.JSONAction(action_type="wait")

        elif action_type == "terminate":
            status = action_args.get("status", "")
            goal_status = "complete" if status == "success" else "infeasible"
            return json_action.JSONAction(action_type="status", goal_status=goal_status)

        print(Colors.error(f"Unknown or malformed mobile_use action: {action_type}"))
        return None

    def _apply_todo_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        todos = arguments.get("todos", [])
        merge = bool(arguments.get("merge", False))
        if not isinstance(todos, list):
            raise ValueError("'todos' must be a list")

        normalized_todos = []
        for todo in todos:
            normalized_todos.append(
                {
                    "id": str(todo["id"]),
                    "content": str(todo["content"]),
                    "status": str(todo["status"]),
                    "priority": str(todo["priority"]),
                }
            )

        if merge:
            todo_map = {item["id"]: copy.deepcopy(item) for item in self.todo_state}
            for todo in normalized_todos:
                todo_map[todo["id"]] = todo
            self.todo_state = list(todo_map.values())
        else:
            self.todo_state = copy.deepcopy(normalized_todos)

        return {
            "tool_name": "write_todos",
            "merge": merge,
            "todo_count": len(self.todo_state),
            "todos": copy.deepcopy(self.todo_state),
        }

    def _apply_memory_tool(self, arguments: dict[str, Any]) -> dict[str, Any]:
        operation = arguments.get("operation")
        memory_id = str(arguments.get("memory_id"))
        description = str(arguments.get("description", ""))
        content = str(arguments.get("content", ""))

        if operation not in {"add", "update", "delete"}:
            raise ValueError(f"Unsupported memory operation: {operation}")

        result = {
            "tool_name": "write_memory",
            "operation": operation,
            "memory_id": memory_id,
            "description": description,
            "content": content,
        }

        if operation == "delete":
            self.memory_state.pop(memory_id, None)
        else:
            self.memory_state[memory_id] = {
                "memory_id": memory_id,
                "description": description,
                "content": content,
            }

        return result

    def _parse_tool_results(
        self, output: str, screen_width: int, screen_height: int
    ) -> tuple[
        json_action.JSONAction | None,
        dict[str, Any],
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[str],
    ]:
        tool_call_blocks = self._extract_tool_call_blocks(output)
        if not tool_call_blocks:
            return None, {}, [], [], ["No tool call found in output"]

        parse_errors: list[str] = []
        if len(tool_call_blocks) > 3:
            parse_errors.append(
                f"More than 3 tool calls found ({len(tool_call_blocks)}); only the first 3 are allowed."
            )
            tool_call_blocks = tool_call_blocks[:3]

        seen_names: set[str] = set()
        parsed_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        mobile_action: json_action.JSONAction | None = None
        parsed_action_dict: dict[str, Any] = {}

        for block in tool_call_blocks:
            try:
                tool_call = self._normalize_tool_call_dict(block)
            except Exception as exc:
                parse_errors.append(f"Failed to parse tool call JSON: {exc}")
                continue

            tool_name = tool_call["name"]
            arguments = tool_call["arguments"]

            if tool_name in seen_names:
                parse_errors.append(f"Tool '{tool_name}' is called more than once.")
                continue
            seen_names.add(tool_name)

            parsed_tool_calls.append({"name": tool_name, "arguments": arguments})

            try:
                if tool_name == "mobile_use":
                    mobile_action = self._parse_mobile_use_arguments(
                        arguments, screen_width, screen_height
                    )
                    if mobile_action is None:
                        parse_errors.append("Failed to parse mobile_use action.")
                    else:
                        parsed_action_dict = self._action_to_dict(mobile_action)
                        tool_results.append(
                            {
                                "tool_name": tool_name,
                                "status": "success",
                                "action_type": mobile_action.action_type,
                            }
                        )
                        self.tool_call_stats["mobile_use"] += 1
                elif tool_name == "write_todos":
                    todo_result = self._apply_todo_tool(arguments)
                    todo_result["status"] = "success"
                    tool_results.append(todo_result)
                    self.tool_call_stats["write_todos"] += 1
                elif tool_name == "write_memory":
                    memory_result = self._apply_memory_tool(arguments)
                    memory_result["status"] = "success"
                    tool_results.append(memory_result)
                    self.tool_call_stats["write_memory"] += 1
                else:
                    parse_errors.append(f"Unsupported tool name: {tool_name}")
            except Exception as exc:
                parse_errors.append(f"Tool '{tool_name}' failed to apply: {exc}")
                tool_results.append(
                    {
                        "tool_name": tool_name,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        if mobile_action is None and parsed_tool_calls:
            mobile_action = json_action.JSONAction(action_type="wait")
            parsed_action_dict = {"action_type": "wait"}

        return (
            mobile_action,
            parsed_action_dict,
            parsed_tool_calls,
            tool_results,
            parse_errors,
        )

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "before_screenshot": None,
            "action_output": None,
            "raw_response": None,
            "parsed_tool_calls": [],
            "tool_results": [],
            "todo_state": [],
            "memory_state": [],
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

        action_history_desc = []
        for hist_item in self.history:
            conclusion = hist_item.get("conclusion", "")
            if conclusion:
                action_history_desc.append(conclusion)
        action_history_desc_str = "\n".join(
            [f"Step {i + 1}: {desc}" for i, desc in enumerate(action_history_desc)]
        )

        user_prompt = (
            f"The user query: {goal}\n"
            f"Task progress:\n{action_history_desc_str if action_history_desc_str else 'None'}\n"
            f"Current todo list: {self._format_todo_state()}\n"
            f"Current memory bank: {self._format_memory_state()}\n"
            "You may update todos, write memory, and/or take one mobile UI action this step."
            "<image>"
        )

        step_data["user_prompt"] = user_prompt
        step_data["system_prompt"] = SYSTEM_PROMPT_V1
        print(Colors.step("User prompt:"))
        print(Colors.info(f"{user_prompt}\n"))

        # Prepare messages for API call
        api_messages = [
            {"role": "system", "content": SYSTEM_PROMPT_V1},
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
            "retry_count": 1,
            "timestamp": api_call_start_time,
            "input_messages": self._sanitize_messages_for_logging(api_messages),
            "model": self.model_name,
            "raw_response": None,
            "parsed_action": None,
            "parsed_tool_calls": [],
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
            model_log_entry["api_call_duration"] = time.time() - api_call_start_time
            response_str = response.choices[0].message.content or ""
            model_log_entry["raw_response"] = response_str

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
        except Exception as exc:
            print(Colors.error(f"Error calling Qwen3-VL v1 model (after retries): {exc}"))
            model_log_entry["error"] = str(exc)
            model_log_entry["api_call_duration"] = time.time() - api_call_start_time
            self.detailed_model_logs.append(model_log_entry)
            step_data["raw_response"] = str(exc)
            step_data["parsed_action"] = {"action_type": "error", "error": str(exc)}
            step_data["action_summary"] = f"Model call failed: {str(exc)[:100]}"
            step_data["conclusion"] = ""
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        print(Colors.important("Qwen3-VL v1 output:"))
        print(Colors.info(f"{response_str}\n"))
        step_data["action_output"] = response_str
        step_data["raw_response"] = response_str

        # Store reasoning_content for Thinking models
        if reasoning_content:
            step_data["reasoning_content"] = reasoning_content

        conclusion = self._extract_conclusion(response_str)
        step_data["conclusion"] = conclusion

        screen_size = self.env.device_screen_size
        (
            action,
            parsed_action_dict,
            parsed_tool_calls,
            tool_results,
            parse_errors,
        ) = self._parse_tool_results(response_str, screen_size[0], screen_size[1])

        step_data["parsed_tool_calls"] = parsed_tool_calls
        step_data["tool_results"] = tool_results
        step_data["todo_state"] = copy.deepcopy(self.todo_state)
        step_data["memory_state"] = self._snapshot_memory_state()
        if parse_errors:
            step_data["parse_errors"] = parse_errors

        for tool_result in tool_results:
            if tool_result.get("tool_name") == "write_memory" and tool_result.get(
                "status"
            ) == "success":
                op = tool_result.get("operation")
                step_data["memory_operation"] = {
                    "operation": op,
                    "memory_id": tool_result.get("memory_id"),
                    "description": tool_result.get("description", ""),
                    "content": tool_result.get("content", ""),
                }
                if op == "add":
                    parsed_action_dict = parsed_action_dict or {"action_type": "wait"}
            elif tool_result.get("tool_name") == "write_todos":
                step_data["todo_operation"] = {
                    "merge": tool_result.get("merge", False),
                    "todos": copy.deepcopy(tool_result.get("todos", [])),
                }

        if action is None:
            print(Colors.error("Failed to parse tools from Qwen3-VL v1 output"))
            step_data["action_summary"] = "Failed to parse tool calls"
            step_data["parsed_action"] = {}
            step_data["action_history_desc"] = action_history_desc + [
                "Failed to parse tool calls"
            ]
            model_log_entry["parsed_action"] = None
            model_log_entry["parsed_tool_calls"] = parsed_tool_calls
            model_log_entry["error"] = "; ".join(parse_errors) or "Failed to parse tools"
            self.detailed_model_logs.append(model_log_entry)
            self.history.append(step_data)
            self._update_qwen3_vl_data(goal, step_num + 1, step_data, {}, conclusion)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["parsed_action"] = parsed_action_dict
        model_log_entry["parsed_action"] = parsed_action_dict
        model_log_entry["parsed_tool_calls"] = parsed_tool_calls
        model_log_entry["success"] = True
        self.detailed_model_logs.append(model_log_entry)

        print(Colors.success(f"Parsed action: {action.action_type}"))
        if getattr(action, "x", None) is not None:
            print(Colors.info(f"  Coordinates: ({action.x}, {action.y})"))

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

        if action.action_type == "answer":
            step_data["action_summary"] = f"Answer: {action.text}"
            try:
                actual_action_coordinates = self.env.execute_action(action)
                step_data["actual_action_coordinates"] = actual_action_coordinates
            except Exception as exc:
                print(Colors.error(f"Error executing answer action: {exc}"))
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

        try:
            actual_action_coordinates = self.env.execute_action(action)
            step_data["actual_action_coordinates"] = actual_action_coordinates
            step_data["action_summary"] = f"{action.action_type}"
            print(Colors.success(f"Executed action: {action.action_type}"))
            time.sleep(2.0)
        except Exception as exc:
            print(Colors.error(f"Error executing action: {exc}"))
            step_data["action_summary"] = f"Error executing {action.action_type}: {exc}"
            action_history_desc.append(
                conclusion if conclusion else step_data["action_summary"]
            )
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            self._update_qwen3_vl_data(
                goal, step_num + 1, step_data, parsed_action_dict, conclusion
            )
            return base_agent.AgentInteractionResult(False, step_data)

        action_history_desc.append(
            conclusion if conclusion else step_data["action_summary"]
        )
        step_data["action_history_desc"] = action_history_desc.copy()

        self.history.append(step_data)
        self._update_qwen3_vl_data(
            goal, step_num + 1, step_data, parsed_action_dict, conclusion
        )
        step_data["qwen3_vl_operations"] = self.qwen3_vl_operations
        step_data["qwen3_vl_step_data"] = self.qwen3_vl_step_data

        return base_agent.AgentInteractionResult(False, step_data)

    def _update_qwen3_vl_data(
        self,
        goal: str,
        _step_id: int,
        _step_data: dict,
        _parsed_action: dict,
        _conclusion: str,
    ):
        self.qwen3_vl_operations = {
            "instruction": goal,
            "episode_id": "episode_0",
            "steps": [],
        }

        for i, hist_item in enumerate(self.history):
            step_info = {
                "step_id": i + 1,
                "image_path": f"./screenshots/step_{i:02d}.png",
                "action": hist_item.get("parsed_action", {}),
                "tool_calls": hist_item.get("parsed_tool_calls", []),
                "tool_results": hist_item.get("tool_results", []),
                "conclusion": hist_item.get("conclusion", ""),
            }
            self.qwen3_vl_operations["steps"].append(step_info)

        self.qwen3_vl_step_data = []
        for i, hist_item in enumerate(self.history):
            step_entry = {
                "step_id": i + 1,
                "screenshot_path": f"./screenshots/step_{i:02d}.png",
                "action_prompt": hist_item.get("user_prompt", ""),
                "system_prompt": hist_item.get("system_prompt", ""),
                "action_output": hist_item.get("raw_response", ""),
                "parsed_action": hist_item.get("parsed_action", {}),
                "parsed_tool_calls": hist_item.get("parsed_tool_calls", []),
                "tool_results": hist_item.get("tool_results", []),
                "todo_state": hist_item.get("todo_state", []),
                "memory_state": hist_item.get("memory_state", []),
                "conclusion": hist_item.get("conclusion", ""),
                "action_history_desc": hist_item.get("action_history_desc", []),
            }
            self.qwen3_vl_step_data.append(step_entry)

    def get_enhanced_log_data(self):
        """
        Get enhanced logging data including detailed model interactions and tool statistics.

        Returns:
          Dictionary containing detailed model logs, token usage, and tool call statistics
        """
        base_data = super().get_enhanced_log_data()
        base_data["tool_call_stats"] = copy.deepcopy(self.tool_call_stats)
        base_data["final_todo_state"] = copy.deepcopy(self.todo_state)
        base_data["final_memory_state"] = self._snapshot_memory_state()
        return base_data
