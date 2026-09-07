"""
Pydantic models for examples, traces, and results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Example(BaseModel):
    """A single evaluation example (conversation + ground truth)."""

    id: str
    filename: str
    channel: str
    date: str
    entity_type: str  # issue, pr, commit, branch, artifact
    entity_id: str  # The actual identifier
    ground_truth_url: str  # Full GitHub URL
    repo: str  # owner/repo
    conversation: str  # Formatted conversation text
    hint_lines: str  # Comma-separated line numbers
    hashed_filename: str = ""
    content_hash: str = ""
    cutoff_date: str = ""  # conversation date + 1 day for time-travel filtering


class StepTrace(BaseModel):
    """A single step in the agent's execution."""

    step: int
    timestamp: float
    response_type: Optional[str] = None  # tool_call, text, error
    tool_call: Optional[Dict[str, Any]] = None
    tool_result: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class RunTrace(BaseModel):
    """Complete trace of a single agent run."""

    example_id: str
    model: str
    tier: int
    budget: int
    steps: List[StepTrace] = Field(default_factory=list)
    plan: Optional[str] = None
    final_answer: Optional[str] = None
    final_confidence: Optional[str] = None
    final_reasoning: Optional[str] = None
    total_tool_calls: int = 0
    total_time_seconds: float = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0          # USD across all turns; 0 if pricing unknown
    cache_read_tokens: int = 0       # cumulative cached input tokens (cache hits)
    cache_creation_tokens: int = 0   # tokens written to cache for next turn
    error: Optional[str] = None


class EvaluationResult(BaseModel):
    """Evaluation result for a single run."""

    correct: bool
    predicted: Optional[str] = None
    expected: str
    num_tool_calls: int = 0
    time_seconds: float = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # True when correct=True only because predicted is a known GitHub duplicate
    # of expected (per data/duplicates.json).
    matched_via_duplicate: bool = False
    duplicate_canonical: Optional[str] = None
    # True when correct=False AND predicted is one of the candidate URLs the
    # offline LLM judge already evaluated against the GT for this example.
    # Distinguishes "agent picked a checked-and-rejected near-miss" from
    # "agent picked something never considered".
    predicted_was_checked_candidate: bool = False
    checked_candidate_verdict: Optional[str] = None  # e.g. "gt_uniquely", "gt_clearly", "neither"


class FailureStage(BaseModel):
    """Classification of failure stage for incorrect answers."""

    stage: str  # TOOL_USE, QUERY_FORMULATION, RESULT_INTERPRETATION, ITERATION_DECISION, ANSWER_SELECTION
    detail: str
    data: Dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    """Complete result for a single example x model x budget run."""

    example_id: str
    model: str
    tier: int
    budget: int
    evaluation: EvaluationResult
    failure_stage: Optional[FailureStage] = None
    trace: RunTrace
    trajectory_analysis: Optional[dict] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class ExperimentSummary(BaseModel):
    """Summary statistics for an experiment configuration."""

    model: str
    tier: int
    budget: int
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    avg_tool_calls: float = 0.0
    avg_time_seconds: float = 0.0
    avg_tokens: float = 0.0
    failure_stages: Dict[str, int] = Field(default_factory=dict)
