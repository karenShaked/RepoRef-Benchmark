"""Repository renames (old owner/repo -> current owner/repo).

GitHub redirects REST URLs of renamed repos, but the search API rejects the
old name (422) and the API returns current-name URLs. Both the scorer and the
tools canonicalize through this map so a renamed repo is the same repo.
Regenerate by GET /repos/{gold repo} and comparing `full_name`.
"""
from __future__ import annotations
import json
import re
from pathlib import Path

_PATH = Path(__file__).resolve().parent / "data" / "repo_redirects.json"
REPO_REDIRECTS = {k.lower(): v for k, v in json.loads(_PATH.read_text()).items()}

_REPO_QUALIFIER = re.compile(r'(?<![\w-])repo:([\w.-]+/[\w.-]+)', re.IGNORECASE)


def canonical_repo(full_name: str) -> str:
    """Current name of a repo; unchanged if it was never renamed."""
    return REPO_REDIRECTS.get(full_name.lower(), full_name)


def canonicalize_query(query: str) -> str:
    """Rewrite repo:old/name qualifiers in a search query to the current name."""
    return _REPO_QUALIFIER.sub(lambda m: f"repo:{canonical_repo(m.group(1))}", query)


def has_repo_qualifier(query: str) -> bool:
    """True if the query already scopes to a repo (repo:x/y, not -repo:x/y)."""
    return bool(_REPO_QUALIFIER.search(query))
