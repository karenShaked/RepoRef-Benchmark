"""
Run one example x one model x one budget.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ..agent.adapters import LiteLLMAdapter
from ..agent.architectures import run_loop
from ..agent.loop import AgentTrace
from ..data.schemas import Example, EvaluationResult, RunResult, RunTrace, StepTrace
from ..scorer import Scorer
from ..tools.registry import ToolRegistry

# One scorer instance (loads redirect_map + duplicates from data/); scores by gold record.
_SCORER = Scorer()


def _load_prompts(config_dir: Path) -> Dict[str, str]:
    """Load prompts from config/prompts.yaml."""
    prompts_file = config_dir / "prompts.yaml"
    if prompts_file.exists():
        with open(prompts_file) as f:
            return yaml.safe_load(f)
    return {}


def _build_user_prompt(
    template: str,
    example: Example,
    budget: int,
) -> str:
    """Fill in the user prompt template."""
    return template.format(
        channel=example.channel,
        date=example.date,
        filtered_csv_messages=example.conversation,
        entity_type=example.entity_type,
        entity_type_upper=example.entity_type.upper(),
        hint_lines=example.hint_lines,
        repo=example.repo,
        budget=budget,
    )


def _trace_to_schema(trace: AgentTrace, tier: int) -> RunTrace:
    """Convert an AgentTrace to the Pydantic RunTrace schema."""
    steps = []
    for s in trace.steps:
        steps.append(StepTrace(
            step=s.get("step", 0),
            timestamp=s.get("timestamp", 0),
            response_type=s.get("response_type"),
            tool_call=s.get("tool_call"),
            tool_result=s.get("tool_result"),
            text=s.get("text"),
            usage=s.get("usage"),
        ))
    # Aggregate cache hits across steps for top-level convenience
    cache_read_total = 0
    cache_creation_total = 0
    for s in trace.steps:
        u = s.get("usage") or {}
        cache_read_total += u.get("cache_read_tokens", 0) or 0
        cache_creation_total += u.get("cache_creation_tokens", 0) or 0

    return RunTrace(
        example_id=trace.example_id,
        model=trace.model,
        tier=tier,
        budget=trace.budget,
        steps=steps,
        plan=trace.plan,
        final_answer=trace.final_answer,
        final_confidence=trace.final_confidence,
        final_reasoning=trace.final_reasoning,
        total_tool_calls=trace.total_tool_calls,
        total_time_seconds=trace.total_time_seconds,
        total_tokens=trace.total_tokens,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        total_cost=getattr(trace, "total_cost", 0.0) or 0.0,
        cache_read_tokens=cache_read_total,
        cache_creation_tokens=cache_creation_total,
        error=trace.error,
    )


def run_single(
    example: Example,
    model_name: str,
    model_config: Dict[str, Any],
    budget: int,
    config_dir: Path,
    github_token: str,
    cache_dir: Optional[Path] = None,
    no_cutoff: bool = False,
    prompt_style: str = "standard",
    thinking_budget: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
) -> RunResult:
    """
    Run one example with one model at one budget.

    Returns a RunResult with evaluation and optional failure stage.
    """
    prompts = _load_prompts(config_dir)
    user_template = prompts.get("user_prompt_template", "{filtered_csv_messages}")
    user_prompt = _build_user_prompt(user_template, example, budget)

    arch_prompts_file = config_dir / "agent_prompts.yaml"
    arch_prompts = {}
    if arch_prompts_file.exists():
        with open(arch_prompts_file) as f:
            arch_prompts = yaml.safe_load(f) or {}

    shared_base = arch_prompts.get("shared_base", prompts.get("system_prompt", ""))
    # Heavy prompt style: append worked examples (ablation)
    heavy_addition = arch_prompts.get("heavy_addition", "") if prompt_style == "heavy" else ""
    system_prompt = shared_base + heavy_addition

    # Override thinking_budget if user passed --thinking-tokens
    if thinking_budget is not None:
        model_config = {**model_config, "thinking_budget": thinking_budget}

    adapter = LiteLLMAdapter(model_config)
    cutoff = example.cutoff_date if not no_cutoff else None
    tools_registry = ToolRegistry(
        github_token=github_token,
        cache_dir=cache_dir,
        cutoff_date=cutoff,
    )

    trace = run_loop(
        adapter=adapter,
        tools_registry=tools_registry,
        conversation=example.conversation,
        example_id=example.id,
        model_name=model_name,
        budget=budget,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_cost_usd=max_cost_usd,
    )

    gold_record = {
        "gold_url": example.ground_truth_url,
        "repo": example.repo,
        "entity_type": example.entity_type,
        "entity_id": example.entity_id,
    }
    evaluation = EvaluationResult(
        correct=_SCORER.match(trace.final_answer or "", gold_record),
        predicted=trace.final_answer,
        expected=example.ground_truth_url,
        num_tool_calls=trace.total_tool_calls,
        time_seconds=trace.total_time_seconds,
        total_tokens=trace.total_tokens,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
    )

    failure_stage = None                # diagnostics (stage/trajectory) not ported
    tier = model_config.get("tier", 0)
    run_trace = _trace_to_schema(trace, tier)
    traj_analysis = None

    return RunResult(
        example_id=example.id,
        model=model_name,
        tier=tier,
        budget=budget,
        evaluation=evaluation,
        failure_stage=failure_stage,
        trace=run_trace,
        trajectory_analysis=traj_analysis,
    )
