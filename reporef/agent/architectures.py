"""
Agent loop — native tool-calling.

A single iterative loop: the model emits text reasoning and/or tool calls per
step; tool results are fed back as observations; iteration continues until the
model calls `submit_answer` or the budget is exhausted. No prompt-level
enforcement of Thought/Action/Observation structure — modern tool-calling
models self-organize reasoning and actions inline.

Debug messages: when DEBUG_MESSAGES_FILE env var is set to a path, each step's
full messages array is appended to that JSONL file. Lets you reconstruct
exactly what the agent saw at every step.
"""

import copy
import json
import os
import time
from typing import Any, Dict, List, Optional

from .adapters import LiteLLMAdapter
from .loop import AgentTrace
from .parsers import extract_github_url
from ..tools.registry import ToolRegistry


def _maybe_log_messages(trace: AgentTrace, step_num: int, messages: List[Dict[str, Any]],
                        phase: str = "before_query"):
    """Append a snapshot of the current messages array to DEBUG_MESSAGES_FILE if set.

    Each line is a JSON object: {example_id, model, step, phase, n_messages, messages}.
    'phase' is 'before_query' (right before model call) or 'after_step' (after appending
    the response + any tool result).
    """
    debug_path = os.environ.get("DEBUG_MESSAGES_FILE", "").strip()
    if not debug_path:
        return
    try:
        record = {
            "example_id": trace.example_id,
            "model": trace.model,
            "step": step_num,
            "phase": phase,
            "n_messages": len(messages),
            "total_chars": sum(len(str(m.get("content", ""))) for m in messages),
            "messages": copy.deepcopy(messages),
        }
        with open(debug_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as e:
        # Never let debug logging crash the run
        pass


def _log_tool_result_size(trace: AgentTrace, tool_name: str, result_str: str):
    """Warn when a single tool result is unusually large (helps spot context-hog calls)."""
    debug_path = os.environ.get("DEBUG_MESSAGES_FILE", "").strip()
    if not debug_path:
        return
    if len(result_str) < 10000:
        return  # only flag big ones
    try:
        record = {
            "example_id": trace.example_id,
            "model": trace.model,
            "warning": "large_tool_result",
            "tool": tool_name,
            "result_chars": len(result_str),
            "result_kb": round(len(result_str) / 1024, 1),
        }
        # Append to a sibling warnings file
        warn_path = debug_path + ".warnings"
        with open(warn_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _record_usage(response, trace: AgentTrace, step_data: dict):
    """Record token usage, cost, and any thinking content from a response."""
    if response.usage:
        step_data["usage"] = response.usage
        trace.total_tokens += response.usage.get("total_tokens", 0)
        trace.prompt_tokens += response.usage.get("prompt_tokens", 0)
        trace.completion_tokens += response.usage.get("completion_tokens", 0)
    trace.total_cost += getattr(response, "cost", 0) or 0
    # Record thinking content (provider-internal reasoning) when exposed.
    # Goes into the trace at step_data["thinking"]; viewable via inspect_agent_messages.
    thinking_content = getattr(response, "thinking", None)
    if thinking_content:
        step_data["thinking"] = thinking_content
        # Also accumulate at trace level for convenience
        if not hasattr(trace, "thinking_log") or trace.thinking_log is None:
            trace.thinking_log = []
        trace.thinking_log.append({
            "step": step_data.get("step"),
            "thinking": thinking_content[:5000],  # cap individual entries
        })


def _handle_tool_call(
    response,
    tools_registry: ToolRegistry,
    messages: List[Dict[str, Any]],
    trace: AgentTrace,
    step_data: dict,
    tool_calls_used: int,
    budget: int,
) -> tuple:
    """
    Handle a tool call response — supports parallel tool calls (Opus 4.7,
    GPT-5, etc. emit multiple tool_calls per assistant turn).

    Returns (tool_calls_used, should_break, budget_exceeded_sent).

    Anthropic invariant: every tool_use block in the assistant message must
    have a matching tool_result in the next user message. We always append
    `response.raw_message` once and then ONE tool-role message per tool_call
    in the same order.
    """
    calls = list(response.tool_calls or [])
    if not calls:
        # Defensive fallback to singular fields
        calls = [{
            "name": response.tool_name,
            "args": response.tool_args or {},
            "id": response.tool_call_id,
        }]

    step_data["response_type"] = "tool_call"
    step_data["text"] = response.text

    # ─────────────────────────────────────────────────────────────────
    # Special case: if ANY parallel call is submit_answer, treat it as
    # the terminal step. Other parallel calls in the same turn are
    # ignored (their tool_use blocks must still be paired — but since
    # we're terminating, we don't need to send the messages back).
    # ─────────────────────────────────────────────────────────────────
    submit_call = next((c for c in calls if c["name"] == "submit_answer"), None)
    if submit_call:
        tool_calls_used += 1
        step_data["tool_call"] = {
            "name": submit_call["name"],
            "arguments": submit_call["args"],
            "call_number": tool_calls_used,
        }
        trace.final_answer = submit_call["args"].get("artifact_url")
        trace.final_confidence = submit_call["args"].get("confidence")
        trace.final_reasoning = submit_call["args"].get("reasoning")
        step_data["tool_result"] = {"status": "submitted"}
        trace.steps.append(step_data)
        return tool_calls_used, True, False

    # ─────────────────────────────────────────────────────────────────
    # Multi-call execution. Append assistant message ONCE, then one
    # tool-role message per call (LiteLLM groups them into a single
    # Anthropic user-with-N-tool_results turn).
    # ─────────────────────────────────────────────────────────────────
    messages.append(response.raw_message)

    per_call_results = []
    budget_exceeded_in_batch = False

    for call in calls:
        tool_calls_used += 1
        call_record = {
            "name": call["name"],
            "arguments": call["args"],
            "call_number": tool_calls_used,
            "id": call["id"],
        }

        if tool_calls_used > budget:
            err = {
                "error": (
                    "BUDGET_EXCEEDED. You have used all your allowed tool calls. "
                    "You MUST now call submit_answer with your best guess."
                )
            }
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": json.dumps(err),
            })
            per_call_results.append({"call": call_record, "result": {"status": "budget_exceeded"}})
            budget_exceeded_in_batch = True
            continue

        result = tools_registry.execute(call["name"], call["args"])
        result_str = json.dumps(result, default=str)
        _log_tool_result_size(trace, call["name"], result_str)
        messages.append({
            "role": "tool",
            "tool_call_id": call["id"],
            "content": result_str,
        })
        per_call_results.append({"call": call_record, "result": result})

    # Single-call: keep legacy step shape so downstream analysis tools (failure-stage
    # classifier, trajectory checks) keep reading step_data["tool_call"] and
    # step_data["tool_result"] directly.
    if len(per_call_results) == 1:
        step_data["tool_call"] = per_call_results[0]["call"]
        step_data["tool_result"] = per_call_results[0]["result"]
    else:
        # Multi-call: store all in step_data; also populate the singular fields
        # with the FIRST call so legacy analysis still works (it'll see the first
        # call as representative).
        step_data["tool_call"] = per_call_results[0]["call"]
        step_data["tool_result"] = per_call_results[0]["result"]
        step_data["parallel_tool_calls"] = per_call_results

    trace.steps.append(step_data)
    return tool_calls_used, False, budget_exceeded_in_batch


def _handle_text_response(
    response,
    messages: List[Dict[str, Any]],
    trace: AgentTrace,
    step_data: dict,
    budget_exceeded_sent: bool,
) -> tuple:
    """
    Handle a text (non-tool-call) response.
    Returns (should_break, budget_exceeded_sent).
    """
    step_data["response_type"] = "text"
    step_data["text"] = response.text
    messages.append({"role": "assistant", "content": response.text})

    # Capture plan from first text response
    if trace.plan is None and response.text:
        trace.plan = response.text

    # Check for embedded URL
    if not trace.final_answer and response.text:
        url = extract_github_url(response.text)
        if url:
            trace.final_answer = url

    if trace.final_answer:
        trace.steps.append(step_data)
        return True, budget_exceeded_sent

    # Budget exceeded retry
    if budget_exceeded_sent:
        trace.steps.append(step_data)
        messages.append({
            "role": "user",
            "content": (
                "You MUST call submit_answer now with your best guess. "
                "No more tool calls are allowed."
            ),
        })
        return False, False  # Only retry once

    # Text-only response without a final answer — append a user nudge so the
    # conversation doesn't end with an assistant turn (newer Anthropic models
    # reject "assistant message prefill"). Tell the model to take action.
    messages.append({
        "role": "user",
        "content": (
            "Please continue: either call a tool to gather more information "
            "or call submit_answer with your final answer."
        ),
    })
    trace.steps.append(step_data)
    return False, budget_exceeded_sent


def run_loop(
    adapter: LiteLLMAdapter,
    tools_registry: ToolRegistry,
    conversation: str,
    example_id: str,
    model_name: str,
    budget: int = 10,
    system_prompt: str = "",
    user_prompt: str = "",
    max_cost_usd: Optional[float] = None,
) -> AgentTrace:
    """
    Native tool-calling agent loop.

    Flow: prompt → {text and/or tool_call} → result → ... → submit_answer.
    Text and tool calls can interleave freely; the model self-organizes
    reasoning. No prompt-enforced Thought/Action/Observation template.
    """
    trace = AgentTrace(example_id=example_id, model=model_name, budget=budget)
    start_time = time.time()

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    tool_calls_used = 0
    budget_exceeded_sent = False
    max_iterations = max(budget + 10, 3)  # at least 3 iterations for budget=0

    # Budget=0: no tools available, model must answer from text alone
    tools_for_query = [] if budget == 0 else tools_registry.get_schemas()

    timeout_seconds = 300  # 5 minutes

    for step_num in range(max_iterations):
        # Timeout check — return trace collected so far
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            trace.error = f"TIMEOUT after {int(elapsed)}s"
            trace.total_tool_calls = tool_calls_used
            trace.total_time_seconds = elapsed
            return trace

        step_data = {
            "step": step_num,
            "timestamp": time.time(),
            "response_type": None,
            "tool_call": None,
            "tool_result": None,
            "text": None,
            "usage": None,
        }

        # Log messages BEFORE the model query (debug only — see DEBUG_MESSAGES_FILE)
        _maybe_log_messages(trace, step_num, messages, phase="before_query")

        try:
            response = adapter.query(
                messages=messages,
                tools=tools_for_query,
            )
        except Exception as e:
            trace.error = str(e)
            step_data["response_type"] = "error"
            step_data["text"] = str(e)
            trace.steps.append(step_data)
            break

        _record_usage(response, trace, step_data)

        # Cost cap: if cumulative cost exceeded, force submit_answer next turn
        # max_cost_usd: 0 or None = unlimited, positive = hard cap in USD
        if max_cost_usd and max_cost_usd > 0 and trace.total_cost > max_cost_usd:
            messages.append({
                "role": "user",
                "content": (
                    f"COST_LIMIT_EXCEEDED. You have spent ${trace.total_cost:.4f} which exceeds "
                    f"the configured cap of ${max_cost_usd:.4f} for this example. "
                    f"You MUST call submit_answer NOW with your best guess. "
                    f"No more tool calls will be processed."
                ),
            })
            # Record this as a synthetic step so the trace shows the cause
            cost_step = {
                "step": step_num + 100,  # tag clearly
                "timestamp": time.time(),
                "response_type": "cost_limit_exceeded",
                "tool_call": None,
                "tool_result": None,
                "text": f"Cost cap ${max_cost_usd:.4f} hit at ${trace.total_cost:.4f}",
                "usage": None,
            }
            trace.steps.append(cost_step)
            # Continue loop — agent should now submit_answer; if not, the
            # next iteration's adapter.query will respond and we keep going
            # until it does (or budget runs out).

        if response.has_tool_call:
            tool_calls_used, should_break, budget_exceeded_sent = _handle_tool_call(
                response, tools_registry, messages, trace, step_data,
                tool_calls_used, budget,
            )
            if should_break:
                break
        else:
            should_break, budget_exceeded_sent = _handle_text_response(
                response, messages, trace, step_data, budget_exceeded_sent,
            )
            if should_break:
                break

    trace.total_tool_calls = tool_calls_used
    trace.total_time_seconds = time.time() - start_time
    return trace
