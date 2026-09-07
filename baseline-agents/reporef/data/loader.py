"""Load RepoRef instances.jsonl + gold into the harness `Example` objects.

The `[REFERENCE] ` marker is already inline in the
anchor message text, which is what the system prompt keys on — so we only flatten the
conversation to the harness's expected text form.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from typing import List

from .schemas import Example


def _flatten(conversation: list) -> str:
    return "\n".join(f'{m["time"]} | {m["user"]}: {m["message"]}' for m in conversation)


def load_examples(instances_path, gold_path) -> List[Example]:
    gold = {}
    for line in Path(gold_path).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = r
    out: List[Example] = []
    for line in Path(instances_path).read_text().splitlines():
        if not line.strip():
            continue
        inst = json.loads(line)
        g = gold[inst["id"]]
        dm = re.search(r"\d{4}-\d{2}-\d{2}", inst["id"])
        out.append(Example(
            id=inst["id"],
            filename=inst["id"] + ".csv",
            channel=inst["id"].split("_")[0],
            date=dm.group(0) if dm else "",
            entity_type=inst["entity_type"],
            entity_id=g["entity_id"],
            ground_truth_url=g["gold_url"],
            repo=inst["repo"],
            conversation=_flatten(inst["conversation"]),
            hint_lines="",
        ))
    return out
