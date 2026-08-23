> **Purpose:** The per-request user prompt for the segmenter. It carries the target entity, channel, a dynamically built `{entity_context}` block, and the formatted `{chat_window}`, then restates the boundary-finding instructions and asks for JSON with `start_line`/`end_line`. 
---

```
TARGET ENTITY: {entity_id}
CHANNEL: {channel}
{entity_context}

CHAT LOG:
---
{chat_window}
---

Find the complete conversation about {entity_id}:
- From the entity message above, trace BACKWARDS through the thread - if a message is a follow-up to an earlier technical message (same topic/same people), include the earlier message too
- Trace FORWARDS to find the LAST TECHNICAL response related to the entity
- STOP before any non-technical off-topic chat (personal updates, greetings, health, weather, etc.) - only include messages about the technical topic
- Do NOT infer topic continuity from proximity alone. A message is related to the entity ONLY if it explicitly references the entity or is a direct reply in the same thread

Output JSON with start_line and end_line.
```
