"""
Core agent loop — model-agnostic, with budget enforcement and planning phase.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .adapters import LiteLLMAdapter, NormalizedResponse
from .parsers import extract_github_url
from ..tools.registry import ToolRegistry


@dataclass
class AgentTrace:
    """Complete trace of a single agent run."""

    example_id: str
    model: str
    budget: int
    steps: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[str] = None
    final_answer: Optional[str] = None
    final_confidence: Optional[str] = None
    final_reasoning: Optional[str] = None
    total_tool_calls: int = 0
    total_time_seconds: float = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    error: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    thinking_log: Optional[List[Dict[str, Any]]] = None  # internal-reasoning content per step


def run_agent(
    adapter: LiteLLMAdapter,
    tools_registry: ToolRegistry,
    conversation: str,
    example_id: str,
    model_name: str,
    budget: int = 10,
    system_prompt: str = "",
    planning_prompt: str = "",
    user_prompt: str = "",
) -> AgentTrace:
    """
    Run one agent session: planning phase → tool-use phase → answer.

    Args:
        adapter: LiteLLM model adapter.
        tools_registry: Tool definitions + execution.
        conversation: The user prompt with the conversation and instructions.
        example_id: Unique ID for this example.
        model_name: Display name for logging.
        budget: Maximum tool calls allowed.
        system_prompt: System prompt text.
        planning_prompt: Planning instructions prepended to conversation.
        user_prompt: The formatted user message (conversation + task).

    Returns:
        AgentTrace with all steps, plan, and final answer.
    """
    trace = AgentTrace(example_id=example_id, model=model_name, budget=budget)
    start_time = time.time()

    # Build the user message with planning prompt prepended
    full_user_message = ""
    if planning_prompt:
        full_user_message += planning_prompt + "\n\n"
    full_user_message += user_prompt

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": full_user_message},
    ]

    # Store the full prompt in the trace so it's visible in the result JSON
    trace.system_prompt = system_prompt
    trace.user_prompt = full_user_message

    tool_calls_used = 0
    budget_exceeded_sent = False

    # Allow extra iterations for text-only responses and budget-exceeded handling
    max_iterations = budget + 10

    for step_num in range(max_iterations):
        step_data: Dict[str, Any] = {
            "step": step_num,
            "timestamp": time.time(),
            "response_type": None,
            "tool_call": None,
            "tool_result": None,
            "text": None,
            "usage": None,
        }

        try:
            response = adapter.query(
                messages=messages,
                tools=tools_registry.get_schemas(),
            )
        except Exception as e:
            trace.error = str(e)
            step_data["response_type"] = "error"
            step_data["text"] = str(e)
            trace.steps.append(step_data)
            break

        if response.usage:
            step_data["usage"] = response.usage
            trace.total_tokens += response.usage.get("total_tokens", 0)
            trace.prompt_tokens += response.usage.get("prompt_tokens", 0)
            trace.completion_tokens += response.usage.get("completion_tokens", 0)

        # ---- Model wants to call a tool ----
        if response.has_tool_call:
            tool_name = response.tool_name
            tool_args = response.tool_args or {}

            step_data["response_type"] = "tool_call"
            step_data["text"] = response.text  # Some models include reasoning text

            # Capture plan from the first text response if it precedes tool calls
            if trace.plan is None and response.text:
                trace.plan = response.text

            # Handle submit_answer
            if tool_name == "submit_answer":
                tool_calls_used += 1
                step_data["tool_call"] = {
                    "name": tool_name,
                    "arguments": tool_args,
                    "call_number": tool_calls_used,
                }
                trace.final_answer = tool_args.get("artifact_url")
                trace.final_confidence = tool_args.get("confidence")
                trace.final_reasoning = tool_args.get("reasoning")
                step_data["tool_result"] = {"status": "submitted"}
                trace.steps.append(step_data)
                break

            tool_calls_used += 1
            step_data["tool_call"] = {
                "name": tool_name,
                "arguments": tool_args,
                "call_number": tool_calls_used,
            }

            # Check budget BEFORE executing (budget = max allowed calls)
            if tool_calls_used > budget:
                # Force the model to answer NOW
                messages.append(response.raw_message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": response.tool_call_id,
                    "content": json.dumps({
                        "error": (
                            "BUDGET_EXCEEDED. You have used all your allowed tool calls. "
                            "You MUST now call submit_answer with your best guess."
                        )
                    }),
                })
                step_data["tool_result"] = {"status": "budget_exceeded"}
                trace.steps.append(step_data)
                budget_exceeded_sent = True
                continue

            # Execute tool
            result = tools_registry.execute(tool_name, tool_args)
            step_data["tool_result"] = result

            # Add to conversation
            messages.append(response.raw_message)
            messages.append({
                "role": "tool",
                "tool_call_id": response.tool_call_id,
                "content": json.dumps(result, default=str),
            })

        # ---- Model gives text response (no tool call) ----
        else:
            step_data["response_type"] = "text"
            step_data["text"] = response.text
            messages.append({"role": "assistant", "content": response.text})

            # Capture plan from the first text-only response
            if trace.plan is None and response.text:
                trace.plan = response.text

            # If model embedded a URL in text, extract it
            if not trace.final_answer and response.text:
                url = extract_github_url(response.text)
                if url:
                    trace.final_answer = url

            # If we already have a final answer or model seems done, stop
            if trace.final_answer:
                trace.steps.append(step_data)
                break

            # If budget was exceeded and model still doesn't submit, try once more
            if budget_exceeded_sent:
                trace.steps.append(step_data)
                # Try one more time to coax an answer
                messages.append({
                    "role": "user",
                    "content": (
                        "You MUST call submit_answer now with your best guess. "
                        "No more tool calls are allowed."
                    ),
                })
                budget_exceeded_sent = False  # Only retry once
                continue

        trace.steps.append(step_data)

    trace.total_tool_calls = tool_calls_used
    trace.total_time_seconds = time.time() - start_time
    return trace
