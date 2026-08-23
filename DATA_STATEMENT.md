# Data Statement

## Source
RepoRef is built from public developer chat on **Gitter** and public artifacts on **GitHub**.
Conversations come from two existing public datasets of open-source developer communication: GitterCom (Parra et al., 2020) and the Gitter issue-discussion dataset (Sahar et al., 2021).

## Construction (summary)
Each instance starts from a chat message that contained a **direct** link to a GitHub artifact.
That link establishes an externally verifiable ground-truth target. The reference-bearing
message is then surgically rewritten to remove the explicit identifier while preserving its
pragmatic role — the rest of the conversation is left unchanged. Instances are retained only
when the target remains recoverable from the conversation plus GitHub evidence, and when it is
distinguishable from strong competing candidates. See the paper for the full pipeline.

## Content and identifiers
Conversations retain the original public Gitter usernames. Messages are real developer
discussion and may contain code, links, and opinions. We do not attempt to identify, profile,
or link individuals to external personal information, and the data is intended solely for
non-commercial research.

## Fields
- `instances.jsonl` — inputs only (conversation + anchor). No answers.
- `gold/*.jsonl` — the answer key, held separately. `original_url` preserves the exact link
  as it appeared in chat when it differs from the current canonical URL (issue↔PR path, or a
  repository that has since been migrated/renamed).
- `redirect_map.json`, `duplicates.json` — used by the scorer to accept migrated-repo and
  closed-as-duplicate artifacts.

## License
Gitter messages are licensed **CC BY-NC-SA**. RepoRef is released under **CC BY-NC-SA** for
non-commercial research use, with attribution to the source datasets. Code in this repository
is under the license in `LICENSE`.

## Known limitations
- Scope is open-source software chat (Gitter/GitHub); other workspaces may differ.
- Targets are limited to issues, pull requests, and commits.
- Instances are constructed from naturally occurring conversations in which the referenced artifact was originally linked explicitly.
