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

# optional "-" (exclusion) and optional quotes: -repo:"a/b"
_REPO_QUALIFIER = re.compile(r'(?<!\w)(-?)repo:(")?([\w.-]+/[\w.-]+)(?(2)")', re.IGNORECASE)


def canonical_repo(full_name: str) -> str:
    """Current name of a repo; unchanged if it was never renamed."""
    return REPO_REDIRECTS.get(full_name.lower(), full_name)


def canonicalize_query(query: str) -> str:
    """Rewrite repo:old/name qualifiers in a search query to the current name."""
    return _REPO_QUALIFIER.sub(
        lambda m: f"{m.group(1)}repo:{canonical_repo(m.group(3))}", query)


def has_repo_qualifier(query: str) -> bool:
    """True if the query already scopes to a repo (repo:x/y, not -repo:x/y)."""
    return any(not m.group(1) for m in _REPO_QUALIFIER.finditer(query))


_URL_REPO = re.compile(r'(github\.com/(?:repos/)?)([^/\s]+/[^/\s]+)', re.IGNORECASE)


def canonicalize_url(url: str) -> str:
    """Rewrite the owner/repo part of a GitHub URL to the current name."""
    return _URL_REPO.sub(lambda m: m.group(1) + canonical_repo(m.group(2)), url, count=1)
