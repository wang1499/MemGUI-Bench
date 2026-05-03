"""Qwen3-VL v2 agent with parallel LLM calls for mobile, todo and memory."""

import base64
import copy
import cv2
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from android_world.agents import base_agent
from android_world.agents import qwen3_vl
from android_world.env import interface
from android_world.env import json_action


Colors = qwen3_vl.Colors

SYSTEM_PROMPT_MOBILE = """
You are a mobile device operation specialist. Your ONLY job is to interact with the mobile UI.

# Tool

You can only call ONE tool: `mobile_use`

<tools>
{
  "type": "function",
  "function": {
    "name": "mobile_use",
    "description": "Use a touchscreen to interact with a mobile device. The screen resolution is normalized to 1000x1000. Use this tool for clicking, typing, swiping, waiting, answering, pressing system buttons, and terminating.",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {"type": "string", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"]},
        "coordinate": {"type": "array"},
        "coordinate2": {"type": "array"},
        "text": {"type": "string"},
        "button": {"type": "string", "enum": ["Back", "Home", "Menu", "Enter"]},
        "status": {"type": "string", "enum": ["success", "failure"]}
      },
      "required": ["action"]
    }
  }
}
</tools>

For each function call, return a JSON object within <tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

1) Thinking: a single short <thinking>...</thinking> block.
2) Tool call: one <tool_call>...</tool_call> block.
3) Conclusion: a short <conclusion>...</conclusion> block.

Rules:
- Output exactly in the order: <thinking>, <tool_call>, <conclusion>.
- Call mobile_use exactly once per step.
- If task is completed, use action=terminate with status="success".
- If task is infeasible, use action=terminate with status="failure".
- **Search Tip**: When search suggestions appear after typing, click the suggestion directly.
"""

SYSTEM_PROMPT_TODO = """
You are a task planning specialist. Your ONLY job is to manage the todo list.

# Tool

You can only call ONE tool: `write_todos`

<tools>
{
  "type": "function",
  "function": {
    "name": "write_todos",
    "description": "Create or update a structured todo list for complex tasks. Use this only when todo tracking helps execution.",
    "parameters": {
      "type": "object",
      "properties": {
        "merge": {"type": "boolean", "description": "If true, merge todos by id; if false, replace the whole todo list."},
        "todos": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "content": {"type": "string"},
              "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]},
              "priority": {"type": "string", "enum": ["high", "medium", "low"]}
            },
            "required": ["id", "content", "status", "priority"]
          }
        }
      },
      "required": ["merge", "todos"]
    }
  }
}
</tools>

For each function call, return a JSON object within <tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

1) Thinking: a single short <thinking>...</thinking> block.
2) Tool call: one <tool_call>...</tool_call> block.

Rules:
- Output exactly in the order: <thinking>, <tool_call>.
- Call write_todos exactly once per step.
- Mark todos as completed immediately after finishing each step.
- Keep exactly one task as in_progress.
- Use clear, actionable task names.
"""

SYSTEM_PROMPT_MEMORY = """
You are a memory management specialist. Your ONLY job is to store and update task memory.

# Tool

You can only call ONE tool: `write_memory`

<tools>
{
  "type": "function",
  "function": {
    "name": "write_memory",
    "description": "Store durable task memory such as discovered facts, constraints, partial findings, credentials, locations, or user preferences for reuse in later steps.",
    "parameters": {
      "type": "object",
      "properties": {
        "operation": {"type": "string", "enum": ["add", "update", "delete"]},
        "memory_id": {"type": "string"},
        "description": {"type": "string"},
        "content": {"type": "string"}
      },
      "required": ["operation", "memory_id"]
    }
  }
}
</tools>

For each function call, return a JSON object within <tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

1) Thinking: a single short <thinking>...</thinking> block.
2) Tool call: one <tool_call>...</tool_call> block.

Rules:
- Output exactly in the order: <thinking>, <tool_call>.
- Call write_memory exactly once per step.
- Store important facts, discovered information, user preferences.
- Use meaningful memory_id for easy retrieval.
- If no new memory needed, use operation="add" with empty content to indicate no change.
"""


class Qwen3VLV2(qwen3_vl.Qwen3VL):
    """Qwen3-VL v2 agent with parallel LLM calls for mobile, todo and memory."""

    def __init__(
        self,
        env: interface.AsyncEnv,
        config: dict[str, Any],
        name: str = "Qwen3VLV2",
    ):
        super().__init__(env, config, name)
        self.todo_state: list[dict[str, Any]] = []
        self.memory_state: dict[str, dict[str, Any]] = {}
        self.tool_call_stats = {
            "mobile_use": 0,
            "write_todos": 0,
            "write_memory": 0,
        }
        self.parallel_model_logs: list[dict[str, Any]] = []

    def reset(self, go_home_on_reset: bool = False):
        super().reset(go_home_on_reset)
        self.todo_state = []
        self.memory_state = {}
        self.tool_call_stats = {
            "mobile_use": 0,
            "write_todos": 0,
            "write_memory": 0,
        }
        self.parallel_model_logs = []

    def _format_todo_state(self) -> str:
        if not self.todo_state:
            return "None"
        return json.dumps(self.todo_state, ensure_ascii=False)

    def _format_memory_state(self) -> str:
        if not self.memory_state:
            return "None"
        ordered_items = [self.memory_state[k] for k in sorted(self.memory_state.keys())]
        return json.dumps(ordered_items, ensure_ascii=False)

    def _extract_tool_call(self, output: str) -> dict[str, Any] | None:
        match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", output, re.DOTALL)
        if not match:
            return None
        try:
            tool_call = json.loads(match.group(1).strip())
            if "name" not in tool_call:
                return None
            if "arguments" not in tool_call:
                tool_call["arguments"] = {
                    k: v for k, v in tool_call.items() if k not in {"name", "arguments"}
                }
            return tool_call
        except json.JSONDecodeError:
            return None

    def _extract_conclusion(self, output: str) -> str:
        match = re.search(r"<conclusion>\s*(.*?)\s*</conclusion>", output, re.DOTALL)
        return match.group(1).strip() if match else ""

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

    def _call_llm(
        self,
        agent_type: str,
        system_prompt: str,
        user_prompt: str,
        screenshot_base64: str,
        step_num: int,
    ) -> dict[str, Any]:
        start_time = time.time()
        log_entry = {
            "agent_type": agent_type,
            "step": step_num,
            "timestamp": start_time,
            "raw_response": None,
            "tool_call": None,
            "success": False,
            "error": None,
            "api_call_duration": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        messages_for_log = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image;base64,<screenshot_placeholder>"},
                    },
                ],
            },
        ]

        messages_for_api = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image;base64,{screenshot_base64}"},
                    },
                ],
            },
        ]

        log_entry["input_messages"] = messages_for_log

        try:
            response = self.client.create_chat_completion(
                model=self.model_name,
                messages=messages_for_api,
            )
            log_entry["api_call_duration"] = time.time() - start_time
            response_str = response.choices[0].message.content or ""
            log_entry["raw_response"] = response_str

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                log_entry["prompt_tokens"] = getattr(usage, "prompt_tokens", 0)
                log_entry["completion_tokens"] = getattr(usage, "completion_tokens", 0)
                log_entry["total_tokens"] = getattr(usage, "total_tokens", 0)

            tool_call = self._extract_tool_call(response_str)
            if tool_call:
                log_entry["tool_call"] = tool_call
                log_entry["success"] = True
            else:
                log_entry["error"] = "No valid tool call found"

        except Exception as exc:
            log_entry["error"] = str(exc)
            log_entry["api_call_duration"] = time.time() - start_time

        return log_entry

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        step_data = {
            "before_screenshot": None,
            "action_output": None,
            "raw_response": None,
            "parsed_tool_calls": [],
            "tool_results": [],
            "todo_state": [],
            "memory_state": [],
            "parallel_logs": [],
        }

        step_num = len(self.history)
        print(Colors.header(f"Step {step_num + 1} [Parallel V2]"))

        state = self.get_post_transition_state()
        step_data["before_screenshot"] = state.pixels.copy()

        _, buffer = cv2.imencode(".png", state.pixels)
        screenshot_base64 = base64.b64encode(buffer).decode("utf-8")
        screenshot_resized = self._resize_screenshot(screenshot_base64)

        action_history_desc = []
        for hist_item in self.history:
            conclusion = hist_item.get("conclusion", "")
            if conclusion:
                action_history_desc.append(conclusion)
        action_history_desc_str = "\n".join(
            [f"Step {i + 1}: {desc}" for i, desc in enumerate(action_history_desc)]
        )

        user_prompt_mobile = (
            f"The user query: {goal}\n"
            f"Task progress:\n{action_history_desc_str if action_history_desc_str else 'None'}\n"
            f"Current todo list: {self._format_todo_state()}\n"
            f"Current memory bank: {self._format_memory_state()}\n"
            "Decide the next mobile UI action."
            "<image>"
        )

        user_prompt_todo = (
            f"The user query: {goal}\n"
            f"Task progress:\n{action_history_desc_str if action_history_desc_str else 'None'}\n"
            f"Current todo list: {self._format_todo_state()}\n"
            f"Current memory bank: {self._format_memory_state()}\n"
            "Update the todo list based on current progress."
            "<image>"
        )

        user_prompt_memory = (
            f"The user query: {goal}\n"
            f"Task progress:\n{action_history_desc_str if action_history_desc_str else 'None'}\n"
            f"Current todo list: {self._format_todo_state()}\n"
            f"Current memory bank: {self._format_memory_state()}\n"
            "Store any important information to memory."
            "<image>"
        )

        print(Colors.step("Calling 3 LLMs in parallel..."))

        parallel_start_time = time.time()
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._call_llm,
                    "mobile",
                    SYSTEM_PROMPT_MOBILE,
                    user_prompt_mobile,
                    screenshot_resized,
                    step_num + 1,
                ): "mobile",
                executor.submit(
                    self._call_llm,
                    "todo",
                    SYSTEM_PROMPT_TODO,
                    user_prompt_todo,
                    screenshot_resized,
                    step_num + 1,
                ): "todo",
                executor.submit(
                    self._call_llm,
                    "memory",
                    SYSTEM_PROMPT_MEMORY,
                    user_prompt_memory,
                    screenshot_resized,
                    step_num + 1,
                ): "memory",
            }

            results = {}
            for future in as_completed(futures):
                agent_type = futures[future]
                try:
                    results[agent_type] = future.result()
                except Exception as exc:
                    results[agent_type] = {
                        "agent_type": agent_type,
                        "success": False,
                        "error": str(exc),
                    }

        parallel_duration = time.time() - parallel_start_time
        print(Colors.success(f"Parallel calls completed in {parallel_duration:.2f}s"))

        step_data["parallel_logs"] = results
        step_data["parallel_duration"] = parallel_duration

        for agent_type, log_entry in results.items():
            if log_entry.get("success"):
                print(Colors.info(f"  [{agent_type}] Success - {log_entry.get('api_call_duration', 0):.2f}s"))
            else:
                print(Colors.error(f"  [{agent_type}] Failed - {log_entry.get('error', 'Unknown error')}"))

        mobile_action: json_action.JSONAction | None = None
        parsed_action_dict: dict[str, Any] = {}
        parsed_tool_calls: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        parse_errors: list[str] = []

        screen_size = self.env.device_screen_size

        mobile_result = results.get("mobile", {})
        if mobile_result.get("success") and mobile_result.get("tool_call"):
            tool_call = mobile_result["tool_call"]
            if tool_call.get("name") == "mobile_use":
                parsed_tool_calls.append({"name": "mobile_use", "arguments": tool_call["arguments"]})
                mobile_action = self._parse_mobile_use_arguments(
                    tool_call["arguments"], screen_size[0], screen_size[1]
                )
                if mobile_action:
                    parsed_action_dict = self._action_to_dict(mobile_action)
                    tool_results.append({
                        "tool_name": "mobile_use",
                        "status": "success",
                        "action_type": mobile_action.action_type,
                    })
                    self.tool_call_stats["mobile_use"] += 1
                else:
                    parse_errors.append("Failed to parse mobile_use action")
            else:
                parse_errors.append(f"Mobile agent returned wrong tool: {tool_call.get('name')}")
        else:
            parse_errors.append(f"Mobile agent failed: {mobile_result.get('error', 'Unknown')}")

        todo_result = results.get("todo", {})
        if todo_result.get("success") and todo_result.get("tool_call"):
            tool_call = todo_result["tool_call"]
            if tool_call.get("name") == "write_todos":
                parsed_tool_calls.append({"name": "write_todos", "arguments": tool_call["arguments"]})
                try:
                    todo_apply_result = self._apply_todo_tool(tool_call["arguments"])
                    todo_apply_result["status"] = "success"
                    tool_results.append(todo_apply_result)
                    self.tool_call_stats["write_todos"] += 1
                except Exception as exc:
                    parse_errors.append(f"Todo tool failed: {exc}")
                    tool_results.append({"tool_name": "write_todos", "status": "error", "error": str(exc)})

        memory_result = results.get("memory", {})
        if memory_result.get("success") and memory_result.get("tool_call"):
            tool_call = memory_result["tool_call"]
            if tool_call.get("name") == "write_memory":
                parsed_tool_calls.append({"name": "write_memory", "arguments": tool_call["arguments"]})
                try:
                    memory_apply_result = self._apply_memory_tool(tool_call["arguments"])
                    memory_apply_result["status"] = "success"
                    tool_results.append(memory_apply_result)
                    self.tool_call_stats["write_memory"] += 1
                except Exception as exc:
                    parse_errors.append(f"Memory tool failed: {exc}")
                    tool_results.append({"tool_name": "write_memory", "status": "error", "error": str(exc)})

        step_data["parsed_tool_calls"] = parsed_tool_calls
        step_data["tool_results"] = tool_results
        step_data["todo_state"] = copy.deepcopy(self.todo_state)
        step_data["memory_state"] = self._snapshot_memory_state()
        if parse_errors:
            step_data["parse_errors"] = parse_errors

        total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in results.values())
        total_completion_tokens = sum(r.get("completion_tokens", 0) for r in results.values())
        self.total_prompt_tokens += total_prompt_tokens
        self.total_completion_tokens += total_completion_tokens

        combined_log = {
            "step": step_num + 1,
            "parallel_duration": parallel_duration,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "mobile": mobile_result,
            "todo": todo_result,
            "memory": memory_result,
            # "parsed_action": parsed_action_dict,
            # "parsed_tool_calls": parsed_tool_calls,
            # "tool_results": tool_results,
        }
        self.detailed_model_logs.append(combined_log)
        self.parallel_model_logs.append(results)

        if mobile_action is None:
            print(Colors.error("Failed to get valid mobile action"))
            step_data["action_summary"] = "Failed to get mobile action"
            step_data["parsed_action"] = {}
            step_data["conclusion"] = ""
            step_data["action_history_desc"] = action_history_desc + ["Failed to get mobile action"]
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        step_data["parsed_action"] = parsed_action_dict
        print(Colors.success(f"Parsed action: {mobile_action.action_type}"))
        if getattr(mobile_action, "x", None) is not None:
            print(Colors.info(f"  Coordinates: ({mobile_action.x}, {mobile_action.y})"))

        conclusion = self._extract_conclusion(mobile_result.get("raw_response", ""))
        step_data["conclusion"] = conclusion

        if mobile_action.action_type == "status":
            step_data["action_summary"] = f"Task completed with status: {mobile_action.goal_status}"
            action_history_desc.append(conclusion if conclusion else step_data["action_summary"])
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(True, step_data)

        if mobile_action.action_type == "answer":
            step_data["action_summary"] = f"Answer: {mobile_action.text}"
            try:
                actual_action_coordinates = self.env.execute_action(mobile_action)
                step_data["actual_action_coordinates"] = actual_action_coordinates
            except Exception as exc:
                print(Colors.error(f"Error executing answer action: {exc}"))
            action_history_desc.append(conclusion if conclusion else step_data["action_summary"])
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        try:
            actual_action_coordinates = self.env.execute_action(mobile_action)
            step_data["actual_action_coordinates"] = actual_action_coordinates
            step_data["action_summary"] = f"{mobile_action.action_type}"
            print(Colors.success(f"Executed action: {mobile_action.action_type}"))
            time.sleep(2.0)
        except Exception as exc:
            print(Colors.error(f"Error executing action: {exc}"))
            step_data["action_summary"] = f"Error executing {mobile_action.action_type}: {exc}"
            action_history_desc.append(conclusion if conclusion else step_data["action_summary"])
            step_data["action_history_desc"] = action_history_desc.copy()
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(False, step_data)

        action_history_desc.append(conclusion if conclusion else step_data["action_summary"])
        step_data["action_history_desc"] = action_history_desc.copy()

        self.history.append(step_data)
        return base_agent.AgentInteractionResult(False, step_data)

    def get_enhanced_log_data(self):
        base_data = super().get_enhanced_log_data()
        base_data["tool_call_stats"] = copy.deepcopy(self.tool_call_stats)
        base_data["final_todo_state"] = copy.deepcopy(self.todo_state)
        base_data["final_memory_state"] = self._snapshot_memory_state()
        base_data["parallel_model_logs"] = copy.deepcopy(self.parallel_model_logs)
        return base_data
