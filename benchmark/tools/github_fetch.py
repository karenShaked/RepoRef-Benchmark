"""
GitHub Fetch — retrieve specific artifacts by owner/repo/id.

Mirrors the GitHub MCP server read-only tools:
issues, PRs, commits, branches, tags, releases, labels, and files.
"""

from typing import Any, Dict, List, Optional

from .github_search import GitHubSearchClient


class GitHubFetchClient:
    """Fetch individual GitHub artifacts (issue, PR, commit, file listing)."""

    def __init__(self, search_client: GitHubSearchClient):
        self._client = search_client

    # ------------------------------------------------------------------
    # Issues
    # ------------------------------------------------------------------

    def get_issue(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        raw = self._client._get(f"/repos/{owner}/{repo}/issues/{number}")
        is_pr = "pull_request" in raw
        return {
            "url": raw.get("html_url", ""),
            "title": raw.get("title", ""),
            "body": (raw.get("body") or "")[:2000],
            "type": "pr" if is_pr else "issue",
            "metadata": {
                "number": raw.get("number"),
                "state": raw.get("state"),
                "user": raw.get("user", {}).get("login", ""),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "labels": [l.get("name", "") for l in raw.get("labels", [])],
                "comments": raw.get("comments", 0),
            },
        }

    def get_issue_comments(
        self,
        owner: str,
        repo: str,
        number: int,
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Get comments on an issue or PR."""
        params = {"per_page": min(per_page, 30)}
        raw = self._client._get(
            f"/repos/{owner}/{repo}/issues/{number}/comments", params
        )
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "user": item.get("user", {}).get("login", ""),
                "body": (item.get("body") or "")[:1000],
                "created_at": item.get("created_at", ""),
                "url": item.get("html_url", ""),
            })
        return {"total_count": len(items), "items": items}

    def list_repo_issues(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        sort: str = "created",
        per_page: int = 10,
    ) -> Dict[str, Any]:
        params = {
            "state": state,
            "sort": sort,
            "per_page": min(per_page, 30),
        }
        raw = self._client._get(f"/repos/{owner}/{repo}/issues", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            is_pr = "pull_request" in item
            items.append({
                "url": item.get("html_url", ""),
                "title": item.get("title", ""),
                "snippet": (item.get("body") or "")[:300],
                "type": "pr" if is_pr else "issue",
                "metadata": {
                    "number": item.get("number"),
                    "state": item.get("state"),
                    "user": item.get("user", {}).get("login", ""),
                    "created_at": item.get("created_at", ""),
                },
            })
        return {"total_count": len(items), "items": items}

    # ------------------------------------------------------------------
    # Pull Requests
    # ------------------------------------------------------------------

    def get_pull_request(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        """Get a specific pull request with merge/review details."""
        raw = self._client._get(f"/repos/{owner}/{repo}/pulls/{number}")
        return {
            "url": raw.get("html_url", ""),
            "title": raw.get("title", ""),
            "body": (raw.get("body") or "")[:2000],
            "type": "pr",
            "metadata": {
                "number": raw.get("number"),
                "state": raw.get("state"),
                "user": raw.get("user", {}).get("login", ""),
                "created_at": raw.get("created_at", ""),
                "updated_at": raw.get("updated_at", ""),
                "merged": raw.get("merged", False),
                "merged_at": raw.get("merged_at"),
                "merged_by": (raw.get("merged_by") or {}).get("login", ""),
                "head_ref": raw.get("head", {}).get("ref", ""),
                "base_ref": raw.get("base", {}).get("ref", ""),
                "commits": raw.get("commits", 0),
                "additions": raw.get("additions", 0),
                "deletions": raw.get("deletions", 0),
                "changed_files": raw.get("changed_files", 0),
                "labels": [l.get("name", "") for l in raw.get("labels", [])],
            },
        }

    def get_pull_request_diff(self, owner: str, repo: str, number: int) -> Dict[str, Any]:
        """Get the diff/files changed in a pull request."""
        raw = self._client._get(f"/repos/{owner}/{repo}/pulls/{number}/files")
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "filename": item.get("filename", ""),
                "status": item.get("status", ""),
                "additions": item.get("additions", 0),
                "deletions": item.get("deletions", 0),
                "changes": item.get("changes", 0),
                "patch": (item.get("patch") or "")[:1000],
            })
        return {"total_count": len(items), "files": items}

    def list_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "all",
        sort: str = "created",
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """List pull requests in a repository."""
        params = {
            "state": state,
            "sort": sort,
            "per_page": min(per_page, 30),
        }
        raw = self._client._get(f"/repos/{owner}/{repo}/pulls", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "url": item.get("html_url", ""),
                "title": item.get("title", ""),
                "snippet": (item.get("body") or "")[:300],
                "type": "pr",
                "metadata": {
                    "number": item.get("number"),
                    "state": item.get("state"),
                    "user": item.get("user", {}).get("login", ""),
                    "created_at": item.get("created_at", ""),
                    "head_ref": item.get("head", {}).get("ref", ""),
                    "base_ref": item.get("base", {}).get("ref", ""),
                },
            })
        return {"total_count": len(items), "items": items}

    def get_pull_request_reviews(
        self, owner: str, repo: str, number: int, per_page: int = 10,
    ) -> Dict[str, Any]:
        """List reviews on a pull request."""
        params = {"per_page": min(per_page, 30)}
        raw = self._client._get(f"/repos/{owner}/{repo}/pulls/{number}/reviews", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "user": item.get("user", {}).get("login", ""),
                "state": item.get("state", ""),
                "body": (item.get("body") or "")[:1000],
                "submitted_at": item.get("submitted_at", ""),
                "html_url": item.get("html_url", ""),
            })
        return {"total_count": len(items), "items": items}

    def get_pull_request_review_comments(
        self, owner: str, repo: str, number: int, per_page: int = 10,
    ) -> Dict[str, Any]:
        """Get line-level review comments on a pull request."""
        params = {"per_page": min(per_page, 30)}
        raw = self._client._get(f"/repos/{owner}/{repo}/pulls/{number}/comments", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "user": item.get("user", {}).get("login", ""),
                "body": (item.get("body") or "")[:1000],
                "path": item.get("path", ""),
                "line": item.get("line"),
                "created_at": item.get("created_at", ""),
                "html_url": item.get("html_url", ""),
            })
        return {"total_count": len(items), "items": items}

    # ------------------------------------------------------------------
    # Commits
    # ------------------------------------------------------------------

    def get_commit(self, owner: str, repo: str, sha: str) -> Dict[str, Any]:
        raw = self._client._get(f"/repos/{owner}/{repo}/commits/{sha}")
        commit = raw.get("commit", {})
        return {
            "url": raw.get("html_url", ""),
            "title": commit.get("message", "").split("\n")[0],
            "body": commit.get("message", "")[:2000],
            "type": "commit",
            "metadata": {
                "sha": raw.get("sha", ""),
                "author": commit.get("author", {}).get("name", ""),
                "date": commit.get("author", {}).get("date", ""),
                "files_changed": len(raw.get("files", [])),
                "stats": raw.get("stats", {}),
            },
        }

    def list_commits(
        self,
        owner: str,
        repo: str,
        sha: str = "",
        author: str = "",
        since: str = "",
        until: str = "",
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """List commits in a repository, optionally filtered."""
        params: Dict[str, Any] = {"per_page": min(per_page, 30)}
        if sha:
            params["sha"] = sha
        if author:
            params["author"] = author
        if since:
            params["since"] = since
        if until:
            params["until"] = until
        raw = self._client._get(f"/repos/{owner}/{repo}/commits", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            commit = item.get("commit", {})
            items.append({
                "url": item.get("html_url", ""),
                "title": commit.get("message", "").split("\n")[0],
                "snippet": commit.get("message", "")[:300],
                "type": "commit",
                "metadata": {
                    "sha": item.get("sha", ""),
                    "author": commit.get("author", {}).get("name", ""),
                    "date": commit.get("author", {}).get("date", ""),
                },
            })
        return {"total_count": len(items), "items": items}

    # ------------------------------------------------------------------
    # Branches
    # ------------------------------------------------------------------

    def list_branches(
        self, owner: str, repo: str, per_page: int = 30
    ) -> Dict[str, Any]:
        """List branches in a repository."""
        params = {"per_page": min(per_page, 100)}
        raw = self._client._get(f"/repos/{owner}/{repo}/branches", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "name": item.get("name", ""),
                "sha": item.get("commit", {}).get("sha", ""),
                "protected": item.get("protected", False),
            })
        return {"total_count": len(items), "items": items}

    # ------------------------------------------------------------------
    # Tags
    # ------------------------------------------------------------------

    def list_tags(
        self, owner: str, repo: str, per_page: int = 30
    ) -> Dict[str, Any]:
        """List tags in a repository."""
        params = {"per_page": min(per_page, 100)}
        raw = self._client._get(f"/repos/{owner}/{repo}/tags", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "name": item.get("name", ""),
                "sha": item.get("commit", {}).get("sha", ""),
                "tarball_url": item.get("tarball_url", ""),
            })
        return {"total_count": len(items), "items": items}

    def get_tag(self, owner: str, repo: str, tag: str) -> Dict[str, Any]:
        """Get details of a specific tag including message and tagger info."""
        raw = self._client._get(f"/repos/{owner}/{repo}/git/ref/tags/{tag}")
        obj = raw.get("object", {})
        tag_type = obj.get("type", "")
        sha = obj.get("sha", "")
        # If annotated tag, fetch the tag object for message/tagger
        if tag_type == "tag" and sha:
            tag_obj = self._client._get(f"/repos/{owner}/{repo}/git/tags/{sha}")
            return {
                "name": tag,
                "sha": sha,
                "message": (tag_obj.get("message") or "")[:1000],
                "tagger": tag_obj.get("tagger", {}).get("name", ""),
                "date": tag_obj.get("tagger", {}).get("date", ""),
                "type": "annotated",
            }
        # Lightweight tag — points directly to a commit
        return {
            "name": tag,
            "sha": sha,
            "type": "lightweight",
        }

    # ------------------------------------------------------------------
    # Releases
    # ------------------------------------------------------------------

    def list_releases(
        self, owner: str, repo: str, per_page: int = 10
    ) -> Dict[str, Any]:
        """List releases in a repository."""
        params = {"per_page": min(per_page, 30)}
        raw = self._client._get(f"/repos/{owner}/{repo}/releases", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "url": item.get("html_url", ""),
                "tag_name": item.get("tag_name", ""),
                "name": item.get("name", ""),
                "body": (item.get("body") or "")[:500],
                "created_at": item.get("created_at", ""),
                "published_at": item.get("published_at", ""),
                "prerelease": item.get("prerelease", False),
                "draft": item.get("draft", False),
                "author": item.get("author", {}).get("login", ""),
            })
        return {"total_count": len(items), "items": items}

    def get_latest_release(self, owner: str, repo: str) -> Dict[str, Any]:
        """Get the latest published release."""
        raw = self._client._get(f"/repos/{owner}/{repo}/releases/latest")
        return {
            "url": raw.get("html_url", ""),
            "tag_name": raw.get("tag_name", ""),
            "name": raw.get("name", ""),
            "body": (raw.get("body") or "")[:2000],
            "created_at": raw.get("created_at", ""),
            "published_at": raw.get("published_at", ""),
            "author": raw.get("author", {}).get("login", ""),
            "assets": [
                {"name": a.get("name", ""), "download_count": a.get("download_count", 0)}
                for a in raw.get("assets", [])
            ],
        }

    def get_release_by_tag(self, owner: str, repo: str, tag: str) -> Dict[str, Any]:
        """Get a release by its tag name."""
        raw = self._client._get(f"/repos/{owner}/{repo}/releases/tags/{tag}")
        return {
            "url": raw.get("html_url", ""),
            "tag_name": raw.get("tag_name", ""),
            "name": raw.get("name", ""),
            "body": (raw.get("body") or "")[:2000],
            "created_at": raw.get("created_at", ""),
            "published_at": raw.get("published_at", ""),
            "author": raw.get("author", {}).get("login", ""),
            "assets": [
                {"name": a.get("name", ""), "download_count": a.get("download_count", 0)}
                for a in raw.get("assets", [])
            ],
        }

    # ------------------------------------------------------------------
    # Labels
    # ------------------------------------------------------------------

    def list_labels(self, owner: str, repo: str, per_page: int = 30) -> Dict[str, Any]:
        """List labels in a repository."""
        params = {"per_page": min(per_page, 100)}
        raw = self._client._get(f"/repos/{owner}/{repo}/labels", params)
        items = []
        for item in raw if isinstance(raw, list) else []:
            items.append({
                "name": item.get("name", ""),
                "description": (item.get("description") or "")[:200],
                "color": item.get("color", ""),
            })
        return {"total_count": len(items), "items": items}

    def get_label(self, owner: str, repo: str, name: str) -> Dict[str, Any]:
        """Get a specific label by name."""
        raw = self._client._get(f"/repos/{owner}/{repo}/labels/{name}")
        return {
            "name": raw.get("name", ""),
            "description": (raw.get("description") or "")[:500],
            "color": raw.get("color", ""),
        }

    # ------------------------------------------------------------------
    # File contents & repository tree
    # ------------------------------------------------------------------

    def get_file_contents(
        self, owner: str, repo: str, path: str, ref: str = ""
    ) -> Dict[str, Any]:
        """Get contents of a file from a repository."""
        params: Dict[str, Any] = {}
        if ref:
            params["ref"] = ref
        raw = self._client._get(f"/repos/{owner}/{repo}/contents/{path}", params)
        if isinstance(raw, list):
            # It's a directory listing
            items = []
            for item in raw:
                items.append({
                    "name": item.get("name", ""),
                    "path": item.get("path", ""),
                    "type": item.get("type", ""),
                    "size": item.get("size", 0),
                })
            return {"type": "directory", "items": items}
        else:
            import base64
            content = ""
            if raw.get("encoding") == "base64" and raw.get("content"):
                try:
                    content = base64.b64decode(raw["content"]).decode("utf-8", errors="replace")
                except Exception:
                    content = "[binary content]"
            return {
                "type": "file",
                "name": raw.get("name", ""),
                "path": raw.get("path", ""),
                "size": raw.get("size", 0),
                "content": content[:5000],
                "sha": raw.get("sha", ""),
                "url": raw.get("html_url", ""),
            }

    def list_repository_tree(
        self, owner: str, repo: str, sha: str = "HEAD", recursive: bool = False
    ) -> Dict[str, Any]:
        """List the file tree of a repository."""
        params: Dict[str, Any] = {}
        if recursive:
            params["recursive"] = "1"
        raw = self._client._get(f"/repos/{owner}/{repo}/git/trees/{sha}", params)
        items = []
        for item in raw.get("tree", []) if isinstance(raw, dict) else []:
            items.append({
                "path": item.get("path", ""),
                "type": item.get("type", ""),
                "size": item.get("size", 0),
                "sha": item.get("sha", ""),
            })
        return {
            "sha": raw.get("sha", "") if isinstance(raw, dict) else "",
            "total_count": len(items),
            "truncated": raw.get("truncated", False) if isinstance(raw, dict) else False,
            "items": items,
        }

