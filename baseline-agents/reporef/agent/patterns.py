"""
Canonical entity-reference regex patterns for GitHub artifact references.

Single source of truth for detecting GitHub artifact references
(issues, PRs, commits, discussions) in chat messages.
"""

import re
from typing import Dict, List, Set

# Regex patterns for different types of GitHub entity references
ENTITY_PATTERNS: Dict[str, List[str]] = {
    'artifact': [
        r'#(\d+)',  # #123 — ambiguous: could be issue or PR
    ],
    'issue': [
        r'issue\s*#?(\d+)',  # issue #123 or issue 123
        r'issue\s+(\d+)',  # issue 123
        r'https?://[^\s]+/(issues|issue)/(\d+)',  # GitHub issue URLs
    ],
    'pr': [
        r'PR\s*#?(\d+)',  # PR #123
        r'pull\s+request\s*#?(\d+)',  # pull request #123
        r'pull\s+request\s+(\d+)',  # pull request 123
        r'https?://[^\s]+/pull/(\d+)',  # GitHub PR URLs
        r'https?://[^\s]+/pulls/(\d+)',  # Alternative PR URLs
    ],
    'commit': [
        r'commit\s+([a-f0-9]{7,40})',  # commit abc123
        r'commit\s+#?([a-f0-9]{7,40})',  # commit #abc123
        r'([a-f0-9]{7,40})\s+commit',  # abc123 commit
        r'https?://[^\s]+/commit/([a-f0-9]{7,40})',  # GitHub commit URLs
    ],
    'discussion': [
        r'discussion\s+#?(\d+)',  # discussion #123
        r'https?://[^\s]+/discussions/(\d+)',  # GitHub discussion URLs
    ],
}

# Known entity type prefixes (used for filename parsing)
ENTITY_TYPES = ('artifact', 'issue', 'pr', 'commit', 'discussion')

# GitHub URL pattern for extracting owner/repo/type/id
GITHUB_URL_PATTERN = re.compile(
    r"https?://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+)/"
    r"(?:issues|pull|commit)/"
    r"(?P<id>[A-Za-z0-9_./%-]+)"
)
