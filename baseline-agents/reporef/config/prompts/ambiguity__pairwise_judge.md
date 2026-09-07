> **Purpose:** Given a masked developer chat and two candidate GitHub artifacts (A and B) shown without URLs/issue numbers, the judge decides which artifact is the more likely referent of the masked implicit reference. It emits a strict JSON verdict on an 8-way A/B scale plus a confidence level, an ambiguity-kind label, verbatim supporting evidence, and one to two sentences of reasoning. This is the core offline ambiguity filter used to detect examples where the gold artifact is not uniquely best (equally-valid / ambiguous pairs).

---

The full prompt sent to the model is the system block followed by a blank line and the assembled user block below.

### System block 

```
Given a masked Slack/Gitter conversation between developers, you must judge which of two GitHub artifacts (Artifact A and Artifact B) is the more likely referent of the implicit reference in the conversation.

Important:
- The conversation has been masked: the actual URL or issue number was replaced by a paraphrased description (the line marked [REFERENCE]).
- Both artifacts are shown WITHOUT URLs or issue numbers.
- You should base your judgment on: title, body content, author, creation date, state, and comments — compared against what the conversation actually discusses.
- Be precise about ambiguity: if both could plausibly be the referent, say so.

Output STRICTLY a JSON object (no other text) following this schema:
{
  "verdict": one of [
    "A_only",                   // chat clearly references A; B is irrelevant or different topic
    "A_clearly_better",         // both topical but chat clearly points to A
    "A_slightly_better",        // both fit; A marginally sharper
    "equally_valid_ambiguous",  // both fit equally — cannot distinguish
    "B_slightly_better",        // both fit; B marginally sharper
    "B_clearly_better",         // both topical but chat clearly points to B
    "B_only",                   // chat clearly references B; A is irrelevant
    "neither_related"           // chat references something else entirely
  ],
  "confidence": "high" | "medium" | "low",
  "ambiguity_kind": one of [
    "n/a",                                          // when verdict is not equally_valid_ambiguous
    "indistinguishable_same_thing",                 // A and B are effectively duplicates of each other
    "indistinguishable_both_plausible_separately",  // distinct artifacts but chat too vague to pick one
    "distinguishable_both_referenced"               // chat actually mentions both (e.g. issue + its PR)
  ],
  "evidence_for_A": "verbatim quote from the chat or 'none'",
  "evidence_for_B": "verbatim quote from the chat or 'none'",
  "reasoning": "1-2 sentences explaining the verdict"
}
```

### User block 

```
Below is the masked conversation followed by two candidate artifacts.

=== MASKED CONVERSATION ===
{input_data["masked_conversation"]}

=== ARTIFACT A ===
title:      {A["title"]}
kind:       {A["kind"]}
author:     {A["author"]}
created_at: {A["createdAt"]}
state:      {A["state"]}
body:
{A["body"]}

comments:
{A["comments_preview"]}

=== ARTIFACT B ===
title:      {B["title"]}
kind:       {B["kind"]}
author:     {B["author"]}
created_at: {B["createdAt"]}
state:      {B["state"]}
body:
{B["body"]}

comments:
{B["comments_preview"]}

=== TASK ===
Which artifact (A or B) is the more likely referent of the implicit reference in the conversation?
Respond with the JSON object only, no other text.
```
