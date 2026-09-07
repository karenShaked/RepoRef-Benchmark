"""
Parsers — extract GitHub URLs and entity identifiers from text.

This is the single source of truth for URL normalization and entity ID
extraction. evaluation/metrics.py imports from here.
"""

import re
from typing import Optional

from .patterns import GITHUB_URL_PATTERN

_GITHUB_URL_PATTERN = GITHUB_URL_PATTERN


def extract_github_url(text: str) -> Optional[str]:
    """Extract the first GitHub artifact URL from text."""
    if not text:
        return None
    match = _GITHUB_URL_PATTERN.search(text)
    return match.group(0) if match else None


def normalize_github_url(url: Optional[str]) -> Optional[str]:
    """
    Normalize a GitHub URL for comparison.

    Strips trailing slashes, lowercases, removes query params and fragments.
    """
    if not url:
        return None
    url = url.strip().rstrip("/").lower()
    # Remove query params and fragments
    url = url.split("?")[0].split("#")[0]
    return url


def extract_entity_id_from_url(url: str) -> Optional[str]:
    """Extract entity identifier (number, SHA, branch name) from a GitHub URL."""
    if not url:
        return None
    match = _GITHUB_URL_PATTERN.search(url)
    if not match:
        return None
    return match.group("id").lower()


def normalize_entity_id(entity_id: str, entity_type: str) -> str:
    """Normalize an entity identifier for comparison."""
    entity_id = entity_id.strip().lower()
    if entity_type in ("issue", "pr"):
        match = re.search(r"\d+", entity_id)
        return match.group(0) if match else entity_id
    elif entity_type == "commit":
        match = re.search(r"[a-f0-9]{7,40}", entity_id)
        return match.group(0) if match else entity_id
    return entity_id


def parse_entity_from_response(response: str, entity_type: str) -> Optional[str]:
    """
    Extract entity identifier from raw LLM response text.

    Supports issue numbers, PR numbers, commit SHAs, branch names.
    """
    if not response:
        return None

    response = response.strip()

    if entity_type in ("issue", "pr"):
        # Extract number
        match = re.search(r"#?(\d+)", response)
        return match.group(1) if match else None

    elif entity_type == "commit":
        # Extract SHA (7-40 hex chars)
        match = re.search(r"\b([a-f0-9]{7,40})\b", response, re.I)
        return match.group(1).lower() if match else None

    elif entity_type == "branch":
        response = response.strip("\"'")
        return response

    return response
