"""Pre-flight credential check — run before any paid model call.

Ported from the sibling `ykwim` project after an incident there: a
present-but-expired GITHUB_TOKEN passed the old "is the env var set" check,
so the run proceeded and the agent burned hundreds of expensive tool-call
retries against a dead GitHub API before anyone noticed — real model
inference cost billed for a run that never had a working search tool.

This module makes both failure classes loud instead of silent:
  1. GITHUB_TOKEN is checked with a real GitHub API call (free, no
     meaningful quota cost) before anything else happens.
  2. The model provider's own key gets a live 1-token completion ping, so
     a missing/invalid/expired provider key fails in one cheap call
     instead of partway through a full split.

Any failure aborts before the real run starts.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import requests

# provider (as used in config/models.yaml) -> required env var name.
# NOTE: "google" maps to GEMINI_API_KEY here (litellm's "gemini/" prefix,
# per this repo's own run.py docstring) — not GOOGLE_API_KEY, which is what
# the sibling ykwim project uses for the same provider name. Don't copy
# that mapping over without checking; the two repos' adapters disagree.
PROVIDER_ENV_VAR = {
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "zai": "Z_AI_KEY",
    "together": "TOGETHER_API_KEY",
}


def _check_github_token(github_token: str) -> str | None:
    """Returns an error string, or None if the token actually works."""
    if not github_token:
        return "GITHUB_TOKEN is not set"
    try:
        r = requests.get(
            "https://api.github.com/rate_limit",
            headers={"Authorization": f"token {github_token}"},
            timeout=10,
        )
    except requests.RequestException as e:
        return f"GITHUB_TOKEN: request to GitHub failed ({e})"
    if r.status_code != 200:
        return (f"GITHUB_TOKEN: GitHub rejected it ({r.status_code} {r.reason}) "
                f"— present but not valid. Regenerate it before continuing.")
    remaining = r.json().get("resources", {}).get("core", {}).get("remaining", "?")
    print(f"  [preflight] GITHUB_TOKEN OK ({remaining} API calls remaining this hour)")
    return None


def _check_provider_key(provider: str, model_config: Dict[str, Any]) -> str | None:
    """Live 1-token completion ping. Returns an error string, or None on success."""
    env_name = PROVIDER_ENV_VAR.get(provider)
    if env_name and not os.environ.get(env_name):
        return f"{env_name}: not set (needed for provider '{provider}')"

    from ..agent.adapters import LiteLLMAdapter

    ping_config = dict(model_config)
    ping_config["max_tokens"] = 1
    try:
        adapter = LiteLLMAdapter(ping_config)
        adapter.query(messages=[{"role": "user", "content": "hi"}], tools=[])
    except Exception as e:
        return f"{provider} ({env_name or 'no env var mapped'}): live ping failed — {e}"
    print(f"  [preflight] {provider} ({model_config.get('model_id')}) API key OK")
    return None


def run_preflight(models: Dict[str, Dict[str, Any]], github_token: str) -> None:
    """Validate every credential this run actually needs. Exits the process
    with a clear message if anything fails — call before any real run."""
    import sys

    print("Pre-flight credential check...")
    failures: list[str] = []

    err = _check_github_token(github_token)
    if err:
        failures.append(err)

    seen_providers: set[str] = set()
    for model_config in models.values():
        provider = model_config["provider"]
        if provider in seen_providers:
            continue
        seen_providers.add(provider)
        err = _check_provider_key(provider, model_config)
        if err:
            failures.append(err)

    if failures:
        print("\nPre-flight check FAILED — refusing to start the run.\n")
        for f in failures:
            print(f"  ✗ {f}")
        print("\nFix the above and re-run. No experiment calls were made.")
        sys.exit(1)

    print("Pre-flight check passed — all credentials are live.\n")
