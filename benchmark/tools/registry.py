"""
Tool registry — defines all available tools and dispatches execution.

Exactly mirrors the GitHub MCP server read-only tools (21 tools + submit_answer):
  Search:   search_issues, search_code, search_pull_requests, search_repositories,
            search_users
  Issues:   issue_read, list_issues
  PRs:      pull_request_read, list_pull_requests
  Commits:  get_commit, list_commits
  Branches: list_branches
  Tags:     list_tags, get_tag
  Releases: list_releases, get_latest_release, get_release_by_tag
  Labels:   list_label, get_label
  Files:    get_file_contents, get_repository_tree
  Special:  submit_answer
"""

import json
from typing import Any, Dict, List, Optional

from .github_search import GitHubSearchClient
from .github_fetch import GitHubFetchClient


# ---------------------------------------------------------------------------
# Tool schema definitions (OpenAI function-calling format)
# Descriptions match the GitHub MCP server exactly.
# ---------------------------------------------------------------------------
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    # ---------------------------------------------------------------
    # Search
    # ---------------------------------------------------------------
    {
        "name": "search_issues",
        "description": (
            "Search for issues in GitHub repositories using issues search "
            "syntax already scoped to is:issue"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query using GitHub issues search syntax",
                },
                "owner": {
                    "type": "string",
                    "description": (
                        "Optional repository owner. If provided with repo, "
                        "only issues for this repository are listed."
                    ),
                },
                "repo": {
                    "type": "string",
                    "description": (
                        "Optional repository name. If provided with owner, "
                        "only issues for this repository are listed."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "comments", "reactions", "reactions-+1", "reactions--1",
                        "reactions-smile", "reactions-thinking_face", "reactions-heart",
                        "reactions-tada", "interactions", "created", "updated",
                    ],
                    "description": "Sort field by number of matches of categories, defaults to best match",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_code",
        "description": (
            "Fast and precise code search across ALL GitHub repositories using "
            "GitHub's native search engine. Best for finding exact symbols, "
            "functions, classes, or specific code patterns."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query using GitHub's powerful code search syntax. "
                        "Examples: 'content:Skill language:Java org:github', "
                        "'NOT is:archived language:Python OR language:go', "
                        "'repo:github/github-mcp-server'. Supports exact matching, "
                        "language filters, path filters, and more."
                    ),
                },
                "sort": {
                    "type": "string",
                    "description": "Sort field ('indexed' only)",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order for results",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_pull_requests",
        "description": (
            "Search for pull requests in GitHub repositories using issues "
            "search syntax already scoped to is:pr"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query using GitHub pull request search syntax",
                },
                "owner": {
                    "type": "string",
                    "description": (
                        "Optional repository owner. If provided with repo, "
                        "only pull requests for this repository are listed."
                    ),
                },
                "repo": {
                    "type": "string",
                    "description": (
                        "Optional repository name. If provided with owner, "
                        "only pull requests for this repository are listed."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": [
                        "comments", "reactions", "reactions-+1", "reactions--1",
                        "reactions-smile", "reactions-thinking_face", "reactions-heart",
                        "reactions-tada", "interactions", "created", "updated",
                    ],
                    "description": "Sort field by number of matches of categories, defaults to best match",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_repositories",
        "description": (
            "Find GitHub repositories by name, description, readme, topics, "
            "or other metadata. Perfect for discovering projects, finding "
            "examples, or locating specific repositories across GitHub."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Repository search query. Examples: "
                        "'machine learning in:name stars:>1000 language:python', "
                        "'topic:react', 'user:facebook'. Supports advanced search "
                        "syntax for precise filtering."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["stars", "forks", "help-wanted-issues", "updated"],
                    "description": "Sort repositories by field, defaults to best match",
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_users",
        "description": (
            "Find GitHub users by username, real name, or other profile "
            "information. Useful for locating developers, contributors, "
            "or team members."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "User search query. Examples: 'john smith', "
                        "'location:seattle', 'followers:>100'. "
                        "Search is automatically scoped to type:user."
                    ),
                },
                "sort": {
                    "type": "string",
                    "enum": ["followers", "repositories", "joined"],
                    "description": (
                        "Sort users by number of followers or repositories, "
                        "or when the person joined GitHub."
                    ),
                },
                "order": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort order",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["query"],
        },
    },
    # ---------------------------------------------------------------
    # Issues
    # ---------------------------------------------------------------
    {
        "name": "issue_read",
        "description": "Get information about a specific issue in a GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["get", "get_comments", "get_sub_issues", "get_labels"],
                    "description": (
                        "The read operation to perform on a single issue.\n"
                        "Options are:\n"
                        "1. get - Get details of a specific issue.\n"
                        "2. get_comments - Get issue comments.\n"
                        "3. get_sub_issues - Get sub-issues of the issue.\n"
                        "4. get_labels - Get labels assigned to the issue.\n"
                    ),
                },
                "owner": {"type": "string", "description": "The owner of the repository"},
                "repo": {"type": "string", "description": "The name of the repository"},
                "issue_number": {"type": "integer", "description": "The number of the issue"},
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
            },
            "required": ["method", "owner", "repo", "issue_number"],
        },
    },
    {
        "name": "list_issues",
        "description": (
            "List issues in a GitHub repository. For pagination, use the "
            "'endCursor' from the previous response's 'pageInfo' in the "
            "'after' parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "state": {
                    "type": "string",
                    "enum": ["OPEN", "CLOSED"],
                    "description": (
                        "Filter by state, by default both open and closed "
                        "issues are returned when not provided"
                    ),
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by labels",
                },
                "since": {
                    "type": "string",
                    "description": "Filter by date (ISO 8601 timestamp)",
                },
                "orderBy": {
                    "type": "string",
                    "enum": ["CREATED_AT", "UPDATED_AT", "COMMENTS"],
                    "description": (
                        "Order issues by field. If provided, the 'direction' "
                        "also needs to be provided."
                    ),
                },
                "direction": {
                    "type": "string",
                    "enum": ["ASC", "DESC"],
                    "description": (
                        "Order direction. If provided, the 'orderBy' "
                        "also needs to be provided."
                    ),
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    # ---------------------------------------------------------------
    # Pull Requests
    # ---------------------------------------------------------------
    {
        "name": "pull_request_read",
        "description": "Get information on a specific pull request in GitHub repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": [
                        "get", "get_diff", "get_status", "get_files",
                        "get_review_comments", "get_reviews", "get_comments",
                    ],
                    "description": (
                        "Action to specify what pull request data needs to be "
                        "retrieved from GitHub.\nPossible options:\n"
                        " 1. get - Get details of a specific pull request.\n"
                        " 2. get_diff - Get the diff of a pull request.\n"
                        " 3. get_status - Get status of a head commit in a pull request.\n"
                        " 4. get_files - Get the list of files changed in a pull request.\n"
                        " 5. get_review_comments - Get review threads on a pull request.\n"
                        " 6. get_reviews - Get the reviews on a pull request.\n"
                        " 7. get_comments - Get comments on a pull request.\n"
                    ),
                },
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "pullNumber": {"type": "integer", "description": "Pull request number"},
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
            },
            "required": ["method", "owner", "repo", "pullNumber"],
        },
    },
    {
        "name": "list_pull_requests",
        "description": (
            "List pull requests in a GitHub repository. If the user specifies "
            "an author, then DO NOT use this tool and use the "
            "search_pull_requests tool instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "state": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by state",
                },
                "sort": {
                    "type": "string",
                    "enum": ["created", "updated", "popularity", "long-running"],
                    "description": "Sort by",
                },
                "direction": {
                    "type": "string",
                    "enum": ["asc", "desc"],
                    "description": "Sort direction",
                },
                "head": {
                    "type": "string",
                    "description": "Filter by head user/org and branch",
                },
                "base": {
                    "type": "string",
                    "description": "Filter by base branch",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    # ---------------------------------------------------------------
    # Commits
    # ---------------------------------------------------------------
    {
        "name": "get_commit",
        "description": "Get details for a commit from a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "sha": {
                    "type": "string",
                    "description": "Commit SHA, branch name, or tag name",
                },
            },
            "required": ["owner", "repo", "sha"],
        },
    },
    {
        "name": "list_commits",
        "description": (
            "Get list of commits of a branch in a GitHub repository. "
            "Returns at least 30 results per page by default, but can "
            "return more if specified using the perPage parameter (up to 100)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "sha": {
                    "type": "string",
                    "description": (
                        "Commit SHA, branch or tag name to list commits of. "
                        "If not provided, uses the default branch of the repository."
                    ),
                },
                "author": {
                    "type": "string",
                    "description": "Author username or email address to filter commits by",
                },
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    # ---------------------------------------------------------------
    # Branches
    # ---------------------------------------------------------------
    {
        "name": "list_branches",
        "description": "List branches in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    # ---------------------------------------------------------------
    # Tags
    # ---------------------------------------------------------------
    {
        "name": "list_tags",
        "description": "List git tags in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_tag",
        "description": "Get details about a specific git tag in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "tag": {"type": "string", "description": "Tag name"},
            },
            "required": ["owner", "repo", "tag"],
        },
    },
    # ---------------------------------------------------------------
    # Releases
    # ---------------------------------------------------------------
    {
        "name": "list_releases",
        "description": "List releases in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "perPage": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "Results per page for pagination (min 1, max 100)",
                },
                "page": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Page number for pagination (min 1)",
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_latest_release",
        "description": "Get the latest release in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_release_by_tag",
        "description": "Get a specific release by its tag name in a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Repository owner"},
                "repo": {"type": "string", "description": "Repository name"},
                "tag": {"type": "string", "description": "Tag name (e.g., 'v1.0.0')"},
            },
            "required": ["owner", "repo", "tag"],
        },
    },
    # ---------------------------------------------------------------
    # Labels
    # ---------------------------------------------------------------
    {
        "name": "list_label",
        "description": "List labels from a repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization name)",
                },
                "repo": {"type": "string", "description": "Repository name"},
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_label",
        "description": "Get a specific label from a repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization name)",
                },
                "repo": {"type": "string", "description": "Repository name"},
                "name": {"type": "string", "description": "Label name."},
            },
            "required": ["owner", "repo", "name"],
        },
    },
    # ---------------------------------------------------------------
    # File contents & repository tree
    # ---------------------------------------------------------------
    {
        "name": "get_file_contents",
        "description": "Get the contents of a file or directory from a GitHub repository",
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization)",
                },
                "repo": {"type": "string", "description": "Repository name"},
                "path": {
                    "type": "string",
                    "description": "Path to file/directory",
                },
                "ref": {
                    "type": "string",
                    "description": (
                        "Accepts optional git refs such as "
                        "`refs/tags/{tag}`, `refs/heads/{branch}` "
                        "or `refs/pull/{pr_number}/head`"
                    ),
                },
                "sha": {
                    "type": "string",
                    "description": (
                        "Accepts optional commit SHA. If specified, "
                        "it will be used instead of ref"
                    ),
                },
            },
            "required": ["owner", "repo"],
        },
    },
    {
        "name": "get_repository_tree",
        "description": (
            "Get the tree structure (files and directories) of a GitHub "
            "repository at a specific ref or SHA"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Repository owner (username or organization)",
                },
                "repo": {"type": "string", "description": "Repository name"},
                "tree_sha": {
                    "type": "string",
                    "description": (
                        "The SHA1 value or ref (branch or tag) name of the tree. "
                        "Defaults to the repository's default branch"
                    ),
                },
                "recursive": {
                    "type": "boolean",
                    "description": (
                        "Setting this parameter to true returns the objects or "
                        "subtrees referenced by the tree. Default is false"
                    ),
                },
            },
            "required": ["owner", "repo"],
        },
    },
    # ---------------------------------------------------------------
    # Answer submission (special — not a GitHub API call)
    # ---------------------------------------------------------------
    {
        "name": "submit_answer",
        "description": (
            "Submit your final answer — the GitHub artifact URL you believe "
            "matches the conversation. Call this when you are confident in your "
            "answer, or when you've exhausted your search options."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_url": {
                    "type": "string",
                    "description": "Full GitHub URL of the artifact (issue, PR, commit, file)",
                },
                "confidence": {
                    "type": "string",
                    "enum": ["high", "medium", "low"],
                    "description": "Your confidence level in this answer",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why you chose this artifact",
                },
            },
            "required": ["artifact_url", "confidence"],
        },
    },
]



# Optional anti-contamination guard: repos an agent must never query (e.g. a mirror
# that holds gold artifacts). Empty by default; maintainers may add (owner, repo) tuples
# and "owner/repo" substrings to block them from search results and fetches.
FORBIDDEN_REPOS: list = []
FORBIDDEN_REPO_PATTERNS: list = []


class ToolRegistry:
    """Manages tool schemas and dispatches tool execution."""

    def __init__(
        self,
        github_token: str,
        cache_dir: Optional[Any] = None,
        cutoff_date: Optional[str] = None,
    ):
        self._search_client = GitHubSearchClient(
            token=github_token, cache_dir=cache_dir
        )
        self._fetch_client = GitHubFetchClient(self._search_client)
        self._cutoff_date = cutoff_date

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return tool schemas in function-calling format."""
        return TOOL_SCHEMAS

    @staticmethod
    def _is_forbidden_repo(owner: str, repo: str) -> bool:
        """Check if owner/repo targets a forbidden benchmark repository."""
        owner_l, repo_l = owner.lower(), repo.lower()
        return any(
            owner_l == fo.lower() and repo_l == fr.lower()
            for fo, fr in FORBIDDEN_REPOS
        )

    @staticmethod
    def _query_targets_forbidden_repo(query: str) -> bool:
        """Check if a search query explicitly targets a forbidden repo."""
        q_lower = query.lower()
        return any(p.lower() in q_lower for p in FORBIDDEN_REPO_PATTERNS)

    @staticmethod
    def _scrub_search_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """Remove items from search results that reference forbidden repos."""
        items = results.get("items", [])
        if not items:
            return results
        filtered = []
        for item in items:
            repo_info = item.get("repository", {})
            if isinstance(repo_info, dict):
                full_name = repo_info.get("full_name", "")
                if any(p.lower() in full_name.lower() for p in FORBIDDEN_REPO_PATTERNS):
                    continue
            # Also check html_url / url fields
            for field in ("html_url", "url", "full_name"):
                val = item.get(field, "")
                if isinstance(val, str) and any(
                    p.lower() in val.lower() for p in FORBIDDEN_REPO_PATTERNS
                ):
                    break
            else:
                filtered.append(item)
                continue
            # If the inner for-loop broke, skip this item
            continue
        results = dict(results)
        results["items"] = filtered
        return results

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result.

        Blocks calls that target forbidden (benchmark dataset) repos and
        scrubs forbidden repo references from search results.
        """
        try:
            # --- Pre-execution: block calls targeting forbidden repos ---
            owner = arguments.get("owner", "")
            repo = arguments.get("repo", "")
            if owner and repo and self._is_forbidden_repo(owner, repo):
                return {
                    "error": (
                        f"Access to {owner}/{repo} is blocked. "
                        f"This repository contains benchmark data. "
                        f"Search the project's actual repository instead."
                    )
                }
            query = arguments.get("query", "")
            if query and self._query_targets_forbidden_repo(query):
                return {
                    "error": (
                        "Search query references a benchmark dataset repository. "
                        "Reformulate your search without referencing it."
                    )
                }

            # ------ Search ------
            if tool_name == "search_issues":
                q = arguments["query"]
                if "is:issue" not in q:
                    q = f"is:issue {q}"
                return self._scrub_search_results(
                    self._search_client.search_issues(
                        query=q,
                        sort=arguments.get("sort", "best-match"),
                        per_page=arguments.get("perPage", 10),
                        cutoff_date=self._cutoff_date,
                    )
                )
            elif tool_name == "search_code":
                return self._scrub_search_results(
                    self._search_client.search_code(
                        query=arguments["query"],
                        sort=arguments.get("sort", "best-match"),
                        per_page=arguments.get("perPage", 10),
                        cutoff_date=self._cutoff_date,
                    )
                )
            elif tool_name == "search_commits":
                # tools.yaml advertises search_commits to the model; the
                # earlier dispatcher omitted this branch, returning
                # "Unknown tool: search_commits" — affecting ~38 runs
                # mostly on commit-type examples.
                return self._scrub_search_results(
                    self._search_client.search_commits(
                        query=arguments["query"],
                        sort=arguments.get("sort", "best-match"),
                        per_page=arguments.get("perPage", 10),
                        cutoff_date=self._cutoff_date,
                    )
                )
            elif tool_name == "search_pull_requests":
                q = arguments["query"]
                if "is:pr" not in q:
                    q = f"is:pr {q}"
                return self._scrub_search_results(
                    self._search_client.search_issues(
                        query=q,
                        sort=arguments.get("sort", "best-match"),
                        per_page=arguments.get("perPage", 10),
                        cutoff_date=self._cutoff_date,
                    )
                )
            elif tool_name == "search_repositories":
                return self._scrub_search_results(
                    self._search_client.search_repositories(
                        query=arguments["query"],
                        sort=arguments.get("sort", "best-match"),
                        per_page=arguments.get("perPage", 10),
                    )
                )
            elif tool_name == "search_users":
                return self._search_client.search_users(
                    query=arguments["query"],
                    sort=arguments.get("sort", "best-match"),
                    per_page=arguments.get("perPage", 10),
                )

            # ------ Issues (bundled: issue_read) ------
            elif tool_name == "issue_read":
                method = arguments["method"]
                o, r, n = arguments["owner"], arguments["repo"], arguments["issue_number"]
                pp = arguments.get("perPage", 10)
                if method == "get":
                    return self._fetch_client.get_issue(owner=o, repo=r, number=n)
                elif method == "get_comments":
                    return self._fetch_client.get_issue_comments(
                        owner=o, repo=r, number=n, per_page=pp,
                    )
                elif method == "get_labels":
                    issue = self._fetch_client.get_issue(owner=o, repo=r, number=n)
                    return {
                        "labels": issue.get("metadata", {}).get("labels", []),
                    }
                elif method == "get_sub_issues":
                    return {"items": [], "note": "Sub-issues not supported via REST API"}
                else:
                    return {"error": f"Unknown issue_read method: {method}"}

            elif tool_name == "list_issues":
                state = arguments.get("state", "all")
                if state == "OPEN":
                    state = "open"
                elif state == "CLOSED":
                    state = "closed"
                sort_map = {
                    "CREATED_AT": "created", "UPDATED_AT": "updated",
                    "COMMENTS": "comments",
                }
                sort = sort_map.get(arguments.get("orderBy", ""), "created")
                return self._fetch_client.list_repo_issues(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    state=state,
                    sort=sort,
                    per_page=arguments.get("perPage", 10),
                )

            # ------ Pull Requests (bundled: pull_request_read) ------
            elif tool_name == "pull_request_read":
                method = arguments["method"]
                o, r = arguments["owner"], arguments["repo"]
                n = arguments["pullNumber"]
                pp = arguments.get("perPage", 10)
                if method == "get":
                    return self._fetch_client.get_pull_request(owner=o, repo=r, number=n)
                elif method in ("get_diff", "get_files"):
                    return self._fetch_client.get_pull_request_diff(owner=o, repo=r, number=n)
                elif method == "get_reviews":
                    return self._fetch_client.get_pull_request_reviews(
                        owner=o, repo=r, number=n, per_page=pp,
                    )
                elif method == "get_review_comments":
                    return self._fetch_client.get_pull_request_review_comments(
                        owner=o, repo=r, number=n, per_page=pp,
                    )
                elif method == "get_comments":
                    return self._fetch_client.get_issue_comments(
                        owner=o, repo=r, number=n, per_page=pp,
                    )
                elif method == "get_status":
                    return {"note": "Commit status check not yet implemented"}
                else:
                    return {"error": f"Unknown pull_request_read method: {method}"}

            elif tool_name == "list_pull_requests":
                return self._fetch_client.list_pull_requests(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    state=arguments.get("state", "all"),
                    sort=arguments.get("sort", "created"),
                    per_page=arguments.get("perPage", 10),
                )

            # ------ Commits ------
            elif tool_name == "get_commit":
                return self._fetch_client.get_commit(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    sha=arguments["sha"],
                )
            elif tool_name == "list_commits":
                return self._fetch_client.list_commits(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    sha=arguments.get("sha", ""),
                    author=arguments.get("author", ""),
                    since=arguments.get("since", ""),
                    until=arguments.get("until", ""),
                    per_page=arguments.get("perPage", 10),
                )

            # ------ Branches ------
            elif tool_name == "list_branches":
                return self._fetch_client.list_branches(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    per_page=arguments.get("perPage", 30),
                )

            # ------ Tags ------
            elif tool_name == "list_tags":
                return self._fetch_client.list_tags(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    per_page=arguments.get("perPage", 30),
                )
            elif tool_name == "get_tag":
                return self._fetch_client.get_tag(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    tag=arguments["tag"],
                )

            # ------ Releases ------
            elif tool_name == "list_releases":
                return self._fetch_client.list_releases(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    per_page=arguments.get("perPage", 10),
                )
            elif tool_name == "get_latest_release":
                return self._fetch_client.get_latest_release(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                )
            elif tool_name == "get_release_by_tag":
                return self._fetch_client.get_release_by_tag(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    tag=arguments["tag"],
                )

            # ------ Labels ------
            elif tool_name == "list_label":
                return self._fetch_client.list_labels(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                )
            elif tool_name == "get_label":
                return self._fetch_client.get_label(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    name=arguments["name"],
                )

            # ------ Files ------
            elif tool_name == "get_file_contents":
                ref = arguments.get("sha") or arguments.get("ref", "")
                return self._fetch_client.get_file_contents(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    path=arguments.get("path", "/"),
                    ref=ref,
                )
            elif tool_name == "get_repository_tree":
                return self._fetch_client.list_repository_tree(
                    owner=arguments["owner"],
                    repo=arguments["repo"],
                    sha=arguments.get("tree_sha", "HEAD"),
                    recursive=arguments.get("recursive", False),
                )

            # ------ Answer ------
            elif tool_name == "submit_answer":
                return {
                    "status": "submitted",
                    "artifact_url": arguments.get("artifact_url", ""),
                    "confidence": arguments.get("confidence", ""),
                    "reasoning": arguments.get("reasoning", ""),
                }
            else:
                return {"error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            return {"error": str(e)}
