"""
GitHub Search API wrapper.

Handles all GitHub Search endpoints with rate limiting,
response normalization, and optional response caching.
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


class GitHubSearchClient:
    """Wraps the GitHub REST API search endpoints."""

    BASE_URL = "https://api.github.com"

    def __init__(
        self,
        token: str,
        max_results: int = 10,
        cache_dir: Optional[Path] = None,
    ):
        self.token = token
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        })
        self._last_request_time: float = 0
        self._cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Rate limiting
    # ------------------------------------------------------------------

    def _rate_limit_wait(self) -> None:
        """GitHub Search API: max 30 req/min. Wait if needed."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 2.0:  # ~30 req/min = 1 req per 2 sec
            time.sleep(2.0 - elapsed)
        self._last_request_time = time.time()

    # ------------------------------------------------------------------
    # Caching (GitHub API responses only — NOT LLM responses)
    # ------------------------------------------------------------------

    def _cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Dict]:
        if not self._cache_dir:
            return None
        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            return json.loads(cache_file.read_text())
        return None

    def _set_cached(self, key: str, data: Dict) -> None:
        if not self._cache_dir:
            return
        cache_file = self._cache_dir / f"{key}.json"
        cache_file.write_text(json.dumps(data))

    # ------------------------------------------------------------------
    # Core request
    # ------------------------------------------------------------------

    def _search(
        self,
        endpoint: str,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> Dict:
        params: Dict[str, Any] = {
            "q": query,
            "per_page": min(per_page, 30),
        }
        if sort != "best-match":
            params["sort"] = sort

        # Check cache
        cache_key = self._cache_key(f"search/{endpoint}", params)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit_wait()

        headers = dict(self.session.headers)
        if extra_headers:
            headers.update(extra_headers)

        response = self.session.get(
            f"{self.BASE_URL}/search/{endpoint}",
            params=params,
            headers=headers,
        )

        if response.status_code == 403:
            # Rate limit exceeded — wait and retry once
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 60)
            time.sleep(min(wait, 120))
            return self._search(endpoint, query, sort, per_page)

        response.raise_for_status()
        data = response.json()
        self._set_cached(cache_key, data)
        return data

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        """Generic GET for non-search endpoints."""
        params = params or {}
        cache_key = self._cache_key(path, params)
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        self._rate_limit_wait()
        response = self.session.get(f"{self.BASE_URL}{path}", params=params)

        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            wait = max(reset_time - time.time(), 60)
            time.sleep(min(wait, 120))
            return self._get(path, params)

        response.raise_for_status()
        data = response.json()
        self._set_cached(cache_key, data)
        return data

    # ------------------------------------------------------------------
    # Search endpoints
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_cutoff(query: str, cutoff_date: Optional[str]) -> str:
        """Append created:<cutoff_date to query if provided."""
        if cutoff_date:
            return f"{query} created:<{cutoff_date}"
        return query

    def search_issues(
        self,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
        cutoff_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        raw = self._search("issues", self._apply_cutoff(query, cutoff_date), sort, per_page)
        return {
            "total_count": raw.get("total_count", 0),
            "items": self._normalize_issues(raw.get("items", [])),
            "raw": raw,
        }

    def search_code(
        self,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
        cutoff_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        # Note: code search doesn't support created: qualifier, so cutoff is not applied
        raw = self._search("code", query, sort, per_page)
        return {
            "total_count": raw.get("total_count", 0),
            "items": self._normalize_code(raw.get("items", [])),
            "raw": raw,
        }

    def search_commits(
        self,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
        cutoff_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        # For commits, use author-date qualifier instead of created
        if cutoff_date:
            query = f"{query} author-date:<{cutoff_date}"
        raw = self._search(
            "commits", query, sort, per_page,
            extra_headers={"Accept": "application/vnd.github.cloak-preview+json"},
        )
        return {
            "total_count": raw.get("total_count", 0),
            "items": self._normalize_commits(raw.get("items", [])),
            "raw": raw,
        }

    def search_repositories(
        self,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
    ) -> Dict[str, Any]:
        raw = self._search("repositories", query, sort, per_page)
        return {
            "total_count": raw.get("total_count", 0),
            "items": self._normalize_repos(raw.get("items", [])),
            "raw": raw,
        }

    def search_users(
        self,
        query: str,
        sort: str = "best-match",
        per_page: int = 10,
    ) -> Dict[str, Any]:
        """Search GitHub users."""
        raw = self._search("users", query, sort, per_page)
        items = []
        for item in raw.get("items", []):
            items.append({
                "url": item.get("html_url", ""),
                "login": item.get("login", ""),
                "type": item.get("type", ""),
                "avatar_url": item.get("avatar_url", ""),
            })
        return {"total_count": raw.get("total_count", 0), "items": items}

    # ------------------------------------------------------------------
    # Normalizers — return consistent [{url, title, snippet, type, metadata}]
    # ------------------------------------------------------------------

    def _normalize_issues(self, items: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            is_pr = "pull_request" in item
            results.append({
                "url": item.get("html_url", ""),
                "title": item.get("title", ""),
                "snippet": (item.get("body") or "")[:300],
                "type": "pr" if is_pr else "issue",
                "metadata": {
                    "number": item.get("number"),
                    "state": item.get("state"),
                    "user": item.get("user", {}).get("login", ""),
                    "created_at": item.get("created_at", ""),
                    "updated_at": item.get("updated_at", ""),
                    "labels": [l.get("name", "") for l in item.get("labels", [])],
                    "comments": item.get("comments", 0),
                },
            })
        return results

    def _normalize_code(self, items: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            results.append({
                "url": item.get("html_url", ""),
                "title": item.get("name", ""),
                "snippet": item.get("path", ""),
                "type": "code",
                "metadata": {
                    "path": item.get("path", ""),
                    "sha": item.get("sha", ""),
                    "repo": item.get("repository", {}).get("full_name", ""),
                    "score": item.get("score", 0),
                },
            })
        return results

    def _normalize_commits(self, items: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            commit = item.get("commit", {})
            results.append({
                "url": item.get("html_url", ""),
                "title": commit.get("message", "").split("\n")[0],
                "snippet": commit.get("message", "")[:300],
                "type": "commit",
                "metadata": {
                    "sha": item.get("sha", ""),
                    "author": commit.get("author", {}).get("name", ""),
                    "date": commit.get("author", {}).get("date", ""),
                    "committer": commit.get("committer", {}).get("name", ""),
                },
            })
        return results

    def _normalize_repos(self, items: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for item in items:
            results.append({
                "url": item.get("html_url", ""),
                "title": item.get("full_name", ""),
                "snippet": (item.get("description") or "")[:300],
                "type": "repository",
                "metadata": {
                    "stars": item.get("stargazers_count", 0),
                    "forks": item.get("forks_count", 0),
                    "language": item.get("language", ""),
                    "updated_at": item.get("updated_at", ""),
                },
            })
        return results
