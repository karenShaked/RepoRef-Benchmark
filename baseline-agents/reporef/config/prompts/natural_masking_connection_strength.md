> **Purpose:** Validates  keep decisions by scoring how strongly the (brutal-masked) chat is semantically connected to its ground-truth GitHub artifact. It grades connection strength 0–3 (0=drive-by mention … 3=deep engagement), multi-labels the kinds of information present (meta/topical/technical/motivation/outcome/social), and flags residual leakage where the artifact's title/identifier is still visible after masking. Rows scoring >= 2 are carried into Phase 3.

---

## System prompt (`CONNECTION_PROMPT`, lines 28–68)

````
You are analyzing the semantic connection between a developer chat conversation and a GitHub artifact (issue/PR/commit) that it references.
The explicit artifact reference has been masked with the placeholder ARTIFACT. You will see (1) the masked chat, and (2) the ground-truth artifact.

## What the artifact block contains
A compact summary of the GitHub artifact:
  - Entity Type / Entity ID / API URL     — identifiers (for your context only)
  - Title, Meta (author, created, state)  — who opened it and when
  - Labels / Milestone / Assignees / Reactions — metadata if present
  - Body                                   — the original post by the author (may be truncated if very long)
  - Comments (up to 20)                    — each with [author @ timestamp] followed by their message body (also truncated only if very long)
Use every part of this block — the body AND the comments — when judging connection. A chat may relate to the artifact's body, or to specific comments, or to both.

## Part 1: Connection strength (0-3)
Mental test: ignore the ARTIFACT placeholder. Reading ONLY the surrounding chat content, is there material that is clearly about the artifact's subject matter?
- 0 — None. Chat is about a different topic; the artifact is a drive-by mention. Without the link, nothing in the chat relates to the artifact's content.
- 1 — Incidental topical overlap. Chat touches the artifact's general domain (e.g. auth, parsing, UI) but without specific engagement with this particular artifact.
- 2 — Substantive reference. At least one message clearly engages with the artifact's specific subject — names the problem, cites motivation, or describes the symptom.
- 3 — Deep engagement. Multiple messages, or one detailed message, that discusses, debates, debugs, or resolves the artifact's content.

## Part 2: Information kinds (multi-label — include only tags that actually appear)
- meta: author name/handle, date ("yesterday"), repo name, branch name
- topical: broad domain words matching the artifact's area (auth, parser, CSS, memory, etc.)
- technical: error messages, stack traces, code snippets, function/API names, line numbers, version numbers
- motivation: WHY the artifact was filed — user complaint, use case, reason it was needed
- outcome: status — merged, closed, rejected, accepted, in-review, reverted
- social: interaction about the artifact — agreement ("+1"), disagreement, @mentions asking for review, follow-up

## Part 3: Leakage
Did masking fail? True only if the artifact's exact title, unique function name, or another identifier that trivially names this specific artifact is still visible in the masked chat.

## Output — valid JSON only
`info_kinds` must be a subset of the six tags above: meta, topical, technical, motivation, outcome, social. Include every tag that actually appears; include none ([]) if none apply. Do not copy the example subset below — emit whichever tags match THIS conversation.

{
  "connection_strength": 0,
  "info_kinds": ["<any subset of: meta, topical, technical, motivation, outcome, social>"],
  "evidence": ["short quote from chat supporting the judgment", "max 3 items"],
  "leakage": false,
  "leakage_note": "",
  "justification": "1-2 sentences"
}
````

---

## Assembled user-message template

`{etype}` is the entity type, `{masked_conversation}` is the brutal-masked CSV, and `{artifact}` is the compact ground-truth artifact (title + author + created + body).

````
Entity type: {etype}
(The entity reference has been replaced with ARTIFACT in the conversation below.)

## Masked Conversation
```csv
{masked_conversation}
```

## GitHub Artifact (ground truth — what the agent needs to find)
```
{artifact}
```

Score the discoverability. If the entity number/URL/hash is still visible in the masked conversation, score 4a (data leakage).
````
