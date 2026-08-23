# RepoRef — Conversational Reference Grounding

**RepoRef** is a benchmark for **Conversational Reference Grounding (CoRG)**: given a
multi-party developer chat where one message refers indirectly to a GitHub artifact
(an issue, pull request, or commit), an agent must recover the exact artifact the
speaker meant — by searching GitHub with tools.

Resolving the reference needs lexical, semantic, and
temporal cues spread across the conversation, using tool-mediated search over the repo.
The direct link was present in the original chat and has been surgically removed; the rest
of the conversation is untouched real developer discussion.

- **400** reference-grounded conversation segments (the core benchmark)
- **92** repositories, **23** Gitter communities
- Every target is a real, verifiable GitHub artifact

---

## Install

```bash
git clone <this-repo> && cd reporef-benchmark
pip install -r requirements.txt
export GITHUB_TOKEN=ghp_...        # read-only PAT, for the GitHub tools
export GEMINI_API_KEY=...          # provider key for the model you run (see config/models.yaml)
```

## Quickstart — reproduce the results

```bash
python -m reporef.run --model gemini-3-flash --split main --budget 10
```

This runs the exact ReAct harness from the paper: a `LiteLLM`-backed agent with the 22
read-only GitHub tools and a `submit_answer` action. It writes `predictions.jsonl` +
per-run traces under `runs/`, and prints accuracy. Available models are the keys in
[`reporef/config/models.yaml`](reporef/config/models.yaml) (Gemini-3-Flash, DeepSeek-V4-Pro,
Claude Sonnet 4.6, GPT-5-mini, Grok-4.1, GLM-5.1, Llama-3.3-70B). Set the matching provider
API key for whichever you run.

Score any predictions file directly:
```bash
python -m reporef.scorer --gold data/gold/main.jsonl --predictions predictions.jsonl
```

---

## Dataset

```
data/
├── main/instances.jsonl     # 400 inputs (NO answers)
├── main/metadata.csv        # per-example repo, entity_type, capability bucket, stats
└── gold/main.jsonl          # answer key (held separate)
```

### Instance format (`instances.jsonl`)
```jsonc
{
  "id": "JSPM_2015-01-07_pr_17",
  "repo": "jspm/jspm.io",
  "entity_type": "pr",                       // "issue" | "pr" | "commit"
  "capability_bucket": "C5",                 // diagnostic bucket (C1..C7); null in extended
  "anchor_message_ids": ["54ad3e9c..."],     // the [REFERENCE] message(s) to resolve
  "conversation": [
    {"message_id": "...", "time": "2015-01-07T11:34:...Z", "user": "geelen",
     "message": "I published a blog post/screencast about JSPM ..."},
    {"message_id": "54ad3e9c...", "time": "...", "user": "guybedford",
     "message": "[REFERENCE] yes, I've already seen the PR that was just opened, ..."}
  ]
}
```
The anchor message is marked inline with a `[REFERENCE]` prefix **and** listed in
`anchor_message_ids`. Speaker handles are the real (public) Gitter usernames.

### Gold format (`gold/main.jsonl`)
```jsonc
{"id": "JSPM_2015-01-07_pr_17",
 "gold_url": "https://github.com/jspm/jspm.io/pull/17",
 "repo": "jspm/jspm.io", "entity_type": "pr", "entity_id": "17"}
```
Some records carry `"original_url"` — the exact link as it appeared in chat — when it
differs from the current canonical URL (issue↔PR path, or a since-migrated repo).

---

## Use your own model or agent

- **Your own model** — add an entry under `models:` in `reporef/config/models.yaml` (any
  provider LiteLLM supports: `provider`, `model_id`, `temperature`, `max_tokens`, optional
  `thinking_budget`/`reasoning_effort`), set its API key, and run `--model <your-key>`.
- **Your own agent/strategy** — build any loop you like on `reporef.tools.registry.ToolRegistry`
  (the 22 read-only GitHub tools) and score with `reporef.scorer.Scorer`. Load examples with
  `reporef.data.loader.load_examples("data/main/instances.jsonl", "data/gold/main.jsonl")`.

Tool calls hit **live GitHub** and cache every response under `reporef/cache/`, so re-runs are
reproducible and offline-replayable; ship a warmed cache to let others reproduce a run without
a token.

---

## Scoring

Exact-match, but forgiving about URL *form* while strict about *which artifact*:

1. **URL match** — normalized equality (case / trailing slash / query insensitive), also
   accepting the `original_url`.
2. **Entity match** — issues/PRs: **owner/repo *and* number** must match (an issue with the
   same number in a different project is **not** accepted); commits: SHA match (7+ char
   prefix), repo-agnostic (a SHA is the same content across forks).
3. **Alias resolution** — repo migrations (`redirect_map.json`) and closed-as-duplicate
   artifacts (`duplicates.json`) are resolved before comparing.

```python
from reporef.scorer import Scorer
s = Scorer("data/gold/main.jsonl")
s.is_correct("JSPM_2015-01-07_pr_17", "https://github.com/jspm/jspm.io/pull/17")  # True
```

## Reference results (budget = 10)

| Model | Success % |
|---|---:|
| Gemini-3-Flash | 67.0 |
| DeepSeek-V4-Pro | 60.2 |
| Grok-4.1-Reasoning | 54.0 |
| GLM-5.1 | 52.0 |
| Claude Sonnet 4.6 | 51.2 |
| GPT-5-mini | 49.0 |

CoRG is far from solved — the best agent leaves roughly one third of references unresolved.

---

## Beyond the core benchmark — extended set

An additional ~1.5k instances live under `data/extras/extended/`
(`instances.jsonl`, `gold.jsonl`, `metadata.csv`, same format). They pass the same
Use them for scale (more training/analysis signal); use `main` for reporting benchmark results.

```bash
python -m reporef.run --model gemini-3-flash --split extended --budget 10
```

---

## License & data

Conversations are public Gitter messages (GitterCom; Sahar et al.) under
**CC BY-NC-SA**, with original handles preserved; the benchmark is released under the
same license for non-commercial research. Targets are public GitHub artifacts. See
[`DATA_STATEMENT.md`](DATA_STATEMENT.md). Code is under the license in [`LICENSE`](LICENSE).
