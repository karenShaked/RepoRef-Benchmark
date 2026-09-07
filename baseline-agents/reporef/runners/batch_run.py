"""
Batch runner — run the full experiment matrix.
"""

import json
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..data.loader import load_examples
from ..data.schemas import Example, ExperimentSummary, RunResult
from .single_run import run_single


def _is_rate_limit_error(trace_error: str) -> bool:
    """Detect transient API errors worth retrying with a backoff: rate limits
    (Anthropic 429, Gemini 429, OpenAI quota) AND overload errors (Gemini 503
    'high demand', generic ServiceUnavailable). Both classes are recoverable
    by waiting; the runner retries up to RATE_LIMIT_MAX_RETRIES times."""
    if not trace_error:
        return False
    s = trace_error.lower()
    return any(tok in s for tok in (
        # Rate limit / quota
        "ratelimiterror", "rate_limit", "rate-limit",
        "429", "resource_exhausted", "quota", "throttl",
        # Transient unavailability / overload (Gemini 503, OpenAI 503/529, etc.)
        "503", "serviceunavailable", "service unavailable",
        "unavailable", "overloaded", "high demand",
    ))


# Litellm errors that reflect the agent's OWN behavior (context blew up, took too
# long, prompt was too big). These ARE valid scored failures and must be saved.
_AGENT_FAULT_TOKENS = (
    "contextwindowexceeded",        # litellm.ContextWindowExceededError + BadRequestError wrap
    "prompt exceeds max length",    # ZaiException prompt-too-long
    "litellm.timeout",              # agent took too long
)


def _is_litellm_error(trace_error: str) -> bool:
    """A litellm-raised exception that is NOT the agent's own fault.

    Infrastructure noise (auth, transient API, generic bad-request) gets skip-saved
    so --resume re-runs the example later. Agent-fault errors — context-window-
    exceeded, prompt-too-long, timeout — DO get saved as scored failures because
    they reflect the agent's behavior."""
    if not trace_error or "litellm" not in trace_error.lower():
        return False
    e = trace_error.lower()
    if any(tok in e for tok in _AGENT_FAULT_TOKENS):
        return False  # agent-fault — save it
    return True


def run_experiment_matrix(
    models: Dict[str, Dict[str, Any]],
    budgets: List[int],
    examples: List[Example],
    config_dir: Path,
    output_dir: Path,
    github_token: str,
    cache_dir: Optional[Path] = None,
    resume: bool = False,
    no_cutoff: bool = False,
    prompt_style: str = "standard",
    thinking_budget: Optional[int] = None,
    max_cost_usd: Optional[float] = None,
) -> List[RunResult]:
    """
    Run the full experiment matrix: models x budgets x examples.

    Results are saved incrementally — one JSON file per run.
    """
    MAX_CONSECUTIVE_RATE_LIMITS = 5
    RATE_LIMIT_BACKOFF_SECONDS = 60  # sleep this long after a 429 before retrying
    RATE_LIMIT_MAX_RETRIES = 2       # per-example retries on rate-limit before giving up

    results: List[RunResult] = []
    total_runs = len(models) * len(budgets) * len(examples)
    completed = 0

    for model_name, model_config in models.items():
        for budget in budgets:
            # "act" is historical — there is only one loop now. See agent/architectures.py.
            run_dir = output_dir / model_name / f"budget_{budget}"
            run_dir.mkdir(parents=True, exist_ok=True)
            consecutive_rate_limits = 0

            for example in examples:
                # Display name = the real entity-id-bearing filename stem
                # (e.g. "Cucumber_2016-05-11_issue_949") rather than the
                # content-hash variant. Used for both progress output and the
                # result JSON filename.
                display_name = (example.filename or "").replace(".csv", "") or example.id
                result_file = run_dir / f"{display_name}.json"

                # Resume support: skip already-completed runs.
                # Also check the legacy content-hash filename for back-compat.
                legacy_file = run_dir / f"{example.id}.json"
                existing_path = result_file if result_file.exists() else (
                    legacy_file if legacy_file.exists() else None
                )
                if resume and existing_path:
                    try:
                        existing = json.loads(existing_path.read_text())
                        result = RunResult(**existing)
                        results.append(result)
                        completed += 1
                        continue
                    except Exception:
                        pass  # Re-run if file is corrupted

                completed += 1
                print(
                    f"  [{completed}/{total_runs}] "
                    f"{model_name} | budget={budget} | {display_name}"
                )

                # Run with rate-limit-aware retry: on 429, sleep and re-run the
                # SAME example up to RATE_LIMIT_MAX_RETRIES times before saving.
                from ..data.schemas import EvaluationResult, RunTrace
                rl_retries_used = 0
                while True:
                    try:
                        result = run_single(
                            example=example,
                            model_name=model_name,
                            model_config=model_config,
                            budget=budget,
                            config_dir=config_dir,
                            github_token=github_token,
                            cache_dir=cache_dir,
                            no_cutoff=no_cutoff,
                            prompt_style=prompt_style,
                            thinking_budget=thinking_budget,
                            max_cost_usd=max_cost_usd,
                        )
                    except Exception as e:
                        result = RunResult(
                            example_id=example.id,
                            model=model_name,
                            tier=model_config.get("tier", 0),
                            budget=budget,
                            evaluation=EvaluationResult(
                                correct=False,
                                expected=example.ground_truth_url,
                            ),
                            trace=RunTrace(
                                example_id=example.id,
                                model=model_name,
                                tier=model_config.get("tier", 0),
                                budget=budget,
                                error=str(e),
                            ),
                        )

                    trace_error = str(result.trace.error or "") if result.trace else ""
                    if _is_rate_limit_error(trace_error) and rl_retries_used < RATE_LIMIT_MAX_RETRIES:
                        rl_retries_used += 1
                        print(
                            f"    RATE LIMIT on {display_name} "
                            f"(attempt {rl_retries_used}/{RATE_LIMIT_MAX_RETRIES}); "
                            f"sleeping {RATE_LIMIT_BACKOFF_SECONDS}s then retrying..."
                        )
                        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                        continue
                    break

                # Detect rate-limit errors (kept for the consecutive-fail abort guard)
                is_rate_limited = _is_rate_limit_error(trace_error)

                if is_rate_limited:
                    consecutive_rate_limits += 1
                    if consecutive_rate_limits >= MAX_CONSECUTIVE_RATE_LIMITS:
                        print(
                            f"\n    *** RATE LIMIT: {consecutive_rate_limits} consecutive 429s — "
                            f"stopping {model_name}/budget_{budget}. "
                            f"Re-run with --resume when quota resets. ***\n"
                        )
                        break
                else:
                    consecutive_rate_limits = 0

                # Skip-save on litellm infrastructure errors (auth, context-window,
                # quota, transient API). These don't reflect the model's reasoning,
                # so we leave no file on disk and --resume will retry them later.
                if _is_litellm_error(trace_error):
                    first_line = trace_error.splitlines()[0] if trace_error else ""
                    print(f"    SKIP (litellm error, not saved):")
                    print(f"      {first_line[:400]}")
                    # Drop the full error next to the would-be result file so it's
                    # diagnosable later without re-running.
                    try:
                        (run_dir / f"_LITELLM_ERROR_{display_name}.txt").write_text(trace_error)
                    except Exception:
                        pass
                    continue

                # Save result (including rate-limited ones so --resume skips them,
                # but they can be purged later)
                result_file.write_text(
                    json.dumps(result.model_dump(), indent=2, default=str)
                )
                results.append(result)

                status = "PASS" if result.evaluation.correct else "FAIL"
                stage = ""
                if result.failure_stage:
                    stage = f" ({result.failure_stage.stage})"
                print(f"    {status}{stage}")

    return results


def summarize_results(results: List[RunResult]) -> List[ExperimentSummary]:
    """Aggregate results into per-model-per-budget summaries."""
    # Group by (model, budget)
    groups: Dict[tuple, List[RunResult]] = {}
    for r in results:
        key = (r.model, r.budget)
        groups.setdefault(key, []).append(r)

    summaries: List[ExperimentSummary] = []
    for (model, budget), group in sorted(groups.items()):
        total = len(group)
        correct = sum(1 for r in group if r.evaluation.correct)
        avg_tc = sum(r.evaluation.num_tool_calls for r in group) / max(total, 1)
        avg_time = sum(r.evaluation.time_seconds for r in group) / max(total, 1)
        avg_tokens = sum(r.evaluation.total_tokens for r in group) / max(total, 1)

        # Count failure stages
        stages: Dict[str, int] = {}
        for r in group:
            if r.failure_stage:
                stages[r.failure_stage.stage] = stages.get(r.failure_stage.stage, 0) + 1

        tier = group[0].tier if group else 0
        summaries.append(ExperimentSummary(
            model=model,
            tier=tier,
            budget=budget,
            total=total,
            correct=correct,
            accuracy=correct / max(total, 1),
            avg_tool_calls=round(avg_tc, 2),
            avg_time_seconds=round(avg_time, 2),
            avg_tokens=round(avg_tokens, 0),
            failure_stages=stages,
        ))

    return summaries
