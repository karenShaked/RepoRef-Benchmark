"""
Model adapters — LiteLLM wrapper with normalized response format.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import litellm


# Fallback price table (USD per 1M tokens). Used when LiteLLM doesn't know
# a (preview) model's pricing. Keys match either model_id substrings or provider.
# Tuple: (substr, input_$/MTok, output_$/MTok, cached_input_$/MTok)
# When cached_input is None, no cache discount is applied (cached tokens billed
# at the full input rate).
_FALLBACK_PRICES: List[tuple] = [
    # Google — verified from ai.google.dev/pricing
    ("gemini-3-flash",        0.50,  3.00,  0.05),   # text/image/video tier
    ("gemini-2.5-flash-lite", 0.10,  0.40,  0.01),
    ("gemini-2.5-flash",      0.30,  2.50,  0.075),  # rough — verify if used
    # Anthropic — verified (5m-cache write at 1.25× input, hits at 0.10× input)
    ("claude-opus-4-7",      15.00, 75.00,  1.50),
    ("claude-sonnet-4-7",     3.00, 15.00,  0.30),
    ("claude-sonnet-4-6",     3.00, 15.00,  0.30),
    ("claude-haiku-4-5",      0.80,  4.00,  0.08),
    # OpenAI — verified from openai.com/api/pricing
    ("gpt-5-mini",            0.25,  2.00,  0.025),
    ("gpt-5",                 5.00, 15.00,  0.50),
    # DeepSeek — official deepseek.com API rate (DeepSeek-Chat tier). The
    # project routes through provider=deepseek (direct API), not OpenRouter.
    ("deepseek",              0.27,  1.10,  0.07),
    # Z.ai — verified from docs.z.ai
    ("glm-5.1",               1.40,  4.40,  0.26),
    ("glm-5",                 1.40,  4.40,  0.26),
    # xAI — grok-4-1-fast rate from models.yaml; cache rate unverified, treat
    # cached tokens as full-priced (conservative)
    ("grok-4-1-fast",         0.20,  0.50,  None),
    ("qwen",                  1.00,  3.00,  None),
    # Provider-level fallbacks (last-resort, no cache discount)
    ("anthropic",             3.00, 15.00,  0.30),
    ("google",                0.30,  2.50,  0.05),
    ("openai",                5.00, 15.00,  0.50),
]


def _add_anthropic_cache_markers(messages: List[Dict[str, Any]],
                                 tool_defs: List[Dict[str, Any]]) -> tuple:
    """Add cache_control markers for Anthropic prompt caching.

    Anthropic allows up to 4 cache breakpoints. We use:
      1. End of system prompt (always cacheable; never changes)
      2. End of tool definitions (always cacheable; never changes)
      3. End of initial user prompt (cacheable from turn 2 onward)
      4. End of LAST tool_result message (cacheable from next turn)

    Returns: (modified_messages, modified_tool_defs)

    Caching system+tools alone saves the cost of resending ~3.5K tokens of
    schemas every turn. Caching turn N's history at turn N+1 saves the
    cost of resending all prior tool results — this is the biggest win
    on long agent runs.
    """
    if not messages:
        return messages, tool_defs

    # Deep-copy only the messages we modify (lighter than full copy)
    new_messages = [dict(m) for m in messages]

    # ── 1. System prompt ──
    if new_messages and new_messages[0].get("role") == "system":
        sys_content = new_messages[0].get("content")
        if isinstance(sys_content, str):
            new_messages[0]["content"] = [{
                "type": "text",
                "text": sys_content,
                "cache_control": {"type": "ephemeral"},
            }]
        elif isinstance(sys_content, list) and sys_content:
            # Already in list form — just add cache_control to last block
            sys_content = list(sys_content)
            sys_content[-1] = {**sys_content[-1], "cache_control": {"type": "ephemeral"}}
            new_messages[0]["content"] = sys_content

    # ── 2. Initial user prompt (the conversation + task) ──
    # First user message is the constant task description — always cacheable
    for i, m in enumerate(new_messages):
        if m.get("role") == "user":
            user_content = m.get("content")
            if isinstance(user_content, str):
                new_messages[i] = {
                    **m,
                    "content": [{
                        "type": "text",
                        "text": user_content,
                        "cache_control": {"type": "ephemeral"},
                    }],
                }
            break  # only first user message

    # ── 3. Last tool result — caches everything before it for the NEXT turn ──
    # Find the last tool message (from agent's perspective, "tool" role)
    last_tool_idx = None
    for i in range(len(new_messages) - 1, -1, -1):
        if new_messages[i].get("role") == "tool":
            last_tool_idx = i
            break
    if last_tool_idx is not None and last_tool_idx > 1:  # skip if it IS the first
        m = new_messages[last_tool_idx]
        content = m.get("content")
        if isinstance(content, str):
            new_messages[last_tool_idx] = {
                **m,
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id", ""),
                    "content": content,
                    "cache_control": {"type": "ephemeral"},
                }],
            }

    # ── 4. Tool definitions: cache the LAST one ──
    new_tool_defs = list(tool_defs)
    if new_tool_defs:
        last = dict(new_tool_defs[-1])
        # cache_control on the function-level dict (LiteLLM forwards to Anthropic)
        last["cache_control"] = {"type": "ephemeral"}
        new_tool_defs[-1] = last

    return new_messages, new_tool_defs


def _fallback_price(model_id: str, provider: str, usage: Dict[str, int]) -> float:
    """Look up a price for a model/provider and compute USD cost from token usage.

    Applies cache_read discount when available. `prompt_tokens` from the API
    is the total input including cached tokens, so cached tokens are split out
    of the full-rate bucket before pricing.
    """
    if not usage:
        return 0.0
    key = (model_id or "").lower()
    prov = (provider or "").lower()
    for entry in _FALLBACK_PRICES:
        substr = entry[0].lower()
        if not (substr in key or substr == prov):
            continue
        in_price = entry[1]
        out_price = entry[2]
        cached_price = entry[3] if len(entry) > 3 else None

        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        cache_read = usage.get("cache_read_tokens", 0) or 0
        # Cache reads are counted inside prompt_tokens — split them out so
        # they're billed at the discounted rate.
        fresh_in = max(0, prompt_tokens - cache_read)

        cost = (fresh_in / 1e6) * in_price + (completion_tokens / 1e6) * out_price
        if cached_price is not None and cache_read > 0:
            cost += (cache_read / 1e6) * cached_price
        else:
            # No cache discount configured → bill cached tokens at full input rate.
            cost += (cache_read / 1e6) * in_price
        return cost
    return 0.0


@dataclass
class NormalizedResponse:
    """Unified response format across all providers."""

    has_tool_call: bool
    # Singular fields = first tool call (kept for back-compat with old code paths)
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    tool_call_id: Optional[str] = None
    # All tool calls — list of {name, args, id}. Length 1+ when has_tool_call.
    tool_calls: List[Dict[str, Any]] = None  # type: ignore
    text: Optional[str] = None
    thinking: Optional[str] = None    # internal-reasoning content (when provider exposes it)
    raw_message: Optional[Dict[str, Any]] = None
    raw_response: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, int]] = None
    cost: float = 0.0   # USD for this single call, computed via litellm pricing tables

    def __post_init__(self):
        if self.tool_calls is None:
            self.tool_calls = []


class LiteLLMAdapter:
    """Provider-agnostic adapter using litellm."""

    # Provider → litellm model prefix
    _PROVIDER_MAP = {
        "anthropic": "anthropic",
        "xai": "xai",
        "google": "gemini",
        "together": "together_ai",
        "openai": "openai",
        "deepseek": "deepseek",
        "dashscope": "dashscope",     # Alibaba Qwen via DashScope endpoint
        "openrouter": "openrouter",   # for Qwen / DeepSeek if going via OpenRouter
        "zai": "zai",                 # Z.ai (GLM models)
    }

    def __init__(self, model_config: Dict[str, Any]):
        self.model = model_config["model_id"]
        self.provider = model_config["provider"]
        self.temperature = model_config.get("temperature", 0)
        self.max_tokens = model_config.get("max_tokens", 4096)
        # thinking_budget: integer for Gemini (max thinking tokens). 0 = disabled.
        # None = let provider auto-decide (default behavior).
        self.thinking_budget = model_config.get("thinking_budget")
        # reasoning_effort: OpenAI o-series knob ("low" / "medium" / "high")
        self.reasoning_effort = model_config.get("reasoning_effort")
        self.litellm_model = self._get_litellm_model()

    def _get_litellm_model(self) -> str:
        prefix = self._PROVIDER_MAP.get(self.provider, self.provider)
        return f"{prefix}/{self.model}"

    def query(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> NormalizedResponse:
        """Send a request to the model and return a normalized response."""
        tool_defs = [{"type": "function", "function": t} for t in tools]

        # Build extra kwargs for thinking-budget / reasoning-effort control
        extra_kwargs: Dict[str, Any] = {}
        if self.thinking_budget is not None:
            if self.provider in ("google", "anthropic"):
                extra_kwargs["thinking"] = {"type": "enabled", "budget_tokens": int(self.thinking_budget)}
            # For other providers, silently ignore — most don't support it
        if self.reasoning_effort:
            # OpenAI o-series and gpt-5 reasoning models take this knob.
            extra_kwargs["reasoning_effort"] = self.reasoning_effort

        # ─────────────────────────────────────────────────────────────────
        # Prompt caching — reduce cost by ~5-10× on multi-turn agent runs.
        # Anthropic: explicit cache_control markers (system prompt, tools,
        #            and last user/tool message). Cached content costs ~10%
        #            of normal input rate.
        # Gemini: implicit caching kicks in automatically when ≥1024 tokens
        #         of context match a previous request. No marker needed.
        # OpenAI: implicit caching for prompts ≥1024 tokens. No marker needed.
        # ─────────────────────────────────────────────────────────────────
        cached_messages = messages
        cached_tool_defs = tool_defs
        if self.provider == "anthropic":
            cached_messages, cached_tool_defs = _add_anthropic_cache_markers(messages, tool_defs)

        # Some reasoning models reject explicit temperature. Anthropic Opus 4.5+
        # deprecate it; OpenAI gpt-5 family only accepts temperature=1 (the default).
        model_lc = self.model.lower()
        skip_temperature = (
            (self.provider == "anthropic"
             and "opus-4-" in model_lc
             and not any(f"opus-4-{v}" in model_lc for v in ("0", "1", "2", "3", "4")))
            or (self.provider == "openai" and model_lc.startswith("gpt-5"))
        )

        completion_kwargs: Dict[str, Any] = dict(
            model=self.litellm_model,
            messages=cached_messages,
            tools=cached_tool_defs,
            max_tokens=self.max_tokens,
            **extra_kwargs,
        )
        if not skip_temperature:
            completion_kwargs["temperature"] = self.temperature

        # Z.ai: .env stores the key as Z_AI_KEY but LiteLLM looks for ZAI_API_KEY.
        if self.provider == "zai":
            z_key = os.environ.get("Z_AI_KEY")
            if z_key:
                completion_kwargs["api_key"] = z_key

        response = litellm.completion(**completion_kwargs)

        choice = response.choices[0]
        msg = choice.message

        # Extract usage info — including cache hits when provider exposes them
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
            # Cache metadata (Anthropic exposes both fields; Gemini exposes
            # cached_content_token_count; OpenAI exposes prompt_tokens_details.cached_tokens)
            cache_read = (
                getattr(response.usage, "cache_read_input_tokens", None)
                or getattr(response.usage, "cached_tokens", None)
                or 0
            )
            cache_creation = getattr(response.usage, "cache_creation_input_tokens", None) or 0
            details = getattr(response.usage, "prompt_tokens_details", None)
            if details and not cache_read:
                cache_read = getattr(details, "cached_tokens", 0) or 0
            if cache_read or cache_creation:
                usage["cache_read_tokens"] = int(cache_read)
                usage["cache_creation_tokens"] = int(cache_creation)

        # Compute per-call cost. Try (1) LiteLLM's response-aware cost, then
        # (2) LiteLLM's per-token cost lookup against its model_cost table,
        # then (3) our manual fallback table for preview models LiteLLM
        # doesn't yet have prices for.
        call_cost = 0.0
        try:
            call_cost = float(litellm.completion_cost(completion_response=response) or 0.0)
        except Exception:
            pass
        if call_cost == 0.0 and usage:
            try:
                # Path 2: dynamic lookup in LiteLLM's model_cost dict
                in_cost, out_cost = litellm.cost_per_token(
                    model=self.litellm_model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                )
                call_cost = float((in_cost or 0.0) + (out_cost or 0.0))
            except Exception:
                pass
        if call_cost == 0.0 and usage:
            # Path 3: manual fallback table for preview models
            call_cost = _fallback_price(self.model, self.provider, usage)

        # Extract thinking / reasoning content (when provider exposes it).
        # LiteLLM normalizes Anthropic / Gemini / OpenAI o1 thinking into
        # message.reasoning_content. Some providers also include thinking blocks
        # in the content array — we check both paths.
        thinking_text = None
        try:
            # Path 1: LiteLLM's normalized reasoning_content field
            rc = getattr(msg, "reasoning_content", None)
            if rc:
                thinking_text = str(rc) if isinstance(rc, str) else json.dumps(rc, default=str)
        except Exception:
            pass
        if thinking_text is None:
            try:
                # Path 2: Anthropic-style thinking blocks in content list
                content = getattr(msg, "content", None) or []
                if isinstance(content, list):
                    blocks = []
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "thinking":
                            blocks.append(c.get("thinking", "") or "")
                        elif hasattr(c, "type") and c.type == "thinking":
                            blocks.append(getattr(c, "thinking", "") or "")
                    if blocks:
                        thinking_text = "\n\n---\n\n".join(b for b in blocks if b)
            except Exception:
                pass

        if msg.tool_calls and len(msg.tool_calls) > 0:
            # Parse all tool_calls — reasoning models (Opus 4.7, GPT-5) often
            # emit multiple in one message to parallelize.
            all_calls = []
            for tc in msg.tool_calls:
                try:
                    tc_args = json.loads(tc.function.arguments)
                except (json.JSONDecodeError, TypeError):
                    tc_args = {"raw": tc.function.arguments}
                all_calls.append({"name": tc.function.name, "args": tc_args, "id": tc.id})

            first = all_calls[0]
            return NormalizedResponse(
                has_tool_call=True,
                tool_name=first["name"],
                tool_args=first["args"],
                tool_call_id=first["id"],
                tool_calls=all_calls,
                text=msg.content,
                thinking=thinking_text,
                raw_message=msg.model_dump(),
                raw_response=response.model_dump(),
                usage=usage,
                cost=call_cost,
            )
        else:
            return NormalizedResponse(
                has_tool_call=False,
                text=msg.content,
                thinking=thinking_text,
                raw_message={"role": "assistant", "content": msg.content},
                raw_response=response.model_dump(),
                usage=usage,
                cost=call_cost,
            )
