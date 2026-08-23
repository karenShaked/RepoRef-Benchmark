"""
Budget controller — enforce tool-call limits.
"""

from typing import List


# Standard budget tiers for experiments
STANDARD_BUDGETS: List[int] = [0, 1, 3, 5, 999]


def get_budgets(budget: int | None = None) -> List[int]:
    """
    Return the list of budgets to evaluate.

    Args:
        budget: If given, return only this budget. Otherwise return all standard budgets.
    """
    if budget is not None:
        return [budget]
    return STANDARD_BUDGETS


def is_budget_exceeded(calls_used: int, budget: int) -> bool:
    """Check if the tool-call budget has been exceeded."""
    return calls_used > budget


def budget_exceeded_message(budget: int) -> str:
    """Return the error message sent to the model when budget is exceeded."""
    return (
        f"BUDGET_EXCEEDED. You have used all {budget} of your allowed tool calls. "
        "You MUST now call submit_answer with your best guess."
    )
