"""RepoRef scorer — forgiving, repo-aware exact match.

A prediction is correct if it resolves to the same GitHub artifact as the gold.
Matching is lenient about *URL form* but strict about *which artifact*:

  L1  URL match ...... normalized equality against gold_url (and the original
                       pre-migration link, when present).
  L2  entity match ... issues/PRs: owner/repo AND number must match (repo-aware,
                       so issue #5831 in a different project is NOT accepted);
                       commits: SHA match (7+ char prefix), repo-agnostic because
                       a commit SHA is globally-unique content shared across forks.
      Renamed repos (repo_redirects.json, e.g. facebook/react -> react/react)
      count as the same repo.
  L3  alias resolve .. repo migrations (redirect_map.json) and closed-as-duplicate
                       artifacts (duplicates.json) are resolved, then L1/L2 re-checked.

Usage:
    from reporef.scorer import Scorer
    s = Scorer("data/gold/main.jsonl")            # + redirect_map.json / duplicates.json auto-loaded
    s.is_correct("JSPM_2015-01-07_pr_17", "https://github.com/jspm/jspm.io/pull/17")  # -> True
    report = s.score({example_id: predicted_url, ...})

CLI:
    python -m reporef.scorer --gold data/gold/main.jsonl --predictions preds.jsonl
    # preds.jsonl: one {"id": ..., "predicted_url": ...} per line
"""
from __future__ import annotations
import json, re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"

_PAT = re.compile(
    r'github\.com/(?:repos/)?([^/\s]+)/([^/\s]+)/(issues?|pulls?|pull|commits?|commit)/([^\s/?#,)\]"\'<>]+)'
)
_TYPE = {"issues": "issue", "issue": "issue", "pull": "pr", "pulls": "pr",
         "commit": "commit", "commits": "commit"}


def normalize(url: str | None) -> str | None:
    if not url:
        return None
    u = url.strip().rstrip("/").lower()
    return u.split("?")[0].split("#")[0]


def _load_repo_redirects(path: Path) -> dict:
    """old owner/repo -> current owner/repo, lowercased (repo_redirects.json)."""
    if not path.exists():
        return {}
    return {k.lower(): v.lower() for k, v in json.loads(path.read_text() or "{}").items()}


def parse_url(url: str | None):
    """-> (owner, repo, type, id) lowercased, or None."""
    m = _PAT.search((url or "").lower())
    if not m:
        return None
    o, r, t, i = m.groups()
    return (o, r, _TYPE.get(t, t), i)


def _commit_prefix_eq(a: str, b: str) -> bool:
    n = min(len(a), len(b))
    return n >= 7 and a[:n] == b[:n]


def _entity_match(pred_parts, gold_repo: str, gold_id: str, gold_type: str,
                  renames: dict | None = None) -> bool:
    if not pred_parts:
        return False
    o, r, _t, i = pred_parts
    if gold_type == "commit":                       # SHA is globally unique -> repo-agnostic
        return _commit_prefix_eq(i, gold_id.lower())
    canon = (lambda x: (renames or {}).get(x, x))
    same_repo = canon(f"{o}/{r}") == canon(gold_repo.lower())   # renamed repo == same repo
    return same_repo and i == gold_id.lower()   # repo-aware


class Scorer:
    def __init__(self, gold_path=None, redirect_map_path=None, duplicates_path=None, data_dir=None):
        self.gold = {}
        if gold_path:
            gold_path = Path(gold_path)
            for line in gold_path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self.gold[r["id"]] = r
            guess = gold_path.parent.parent             # data/gold/x.jsonl -> data/
            if data_dir is None and (guess / "redirect_map.json").exists():
                data_dir = guess
        if data_dir is None:                        # e.g. extras/extended/gold.jsonl
            data_dir = _DATA_DIR
        data_dir = Path(data_dir)
        self.renames = _load_repo_redirects(data_dir / "repo_redirects.json")
        self.aliases = {}
        for p in (redirect_map_path or data_dir / "redirect_map.json",
                  duplicates_path or data_dir / "duplicates.json"):
            p = Path(p)
            if p.exists():
                for k, v in json.loads(p.read_text() or "{}").items():
                    self.aliases[self._norm(k)] = self._norm(v)

    def _norm(self, url: str | None) -> str | None:
        """normalize() + renamed repos rewritten to their current name."""
        n = normalize(url)
        pp = parse_url(n)
        if not pp:
            return n
        old = f"{pp[0]}/{pp[1]}"
        new = self.renames.get(old)
        return n.replace(f"github.com/{old}/", f"github.com/{new}/", 1) if new else n

    def _alias(self, url: str | None):
        n = self._norm(url)
        seen = set()
        while n in self.aliases and n not in seen:   # follow chains, guard cycles
            seen.add(n); n = self.aliases[n]
        return n

    def _match_kind(self, predicted: str, g: dict) -> str | None:
        """Return how it matched: 'exact' (L1/L2), 'dup' (only via alias/redirect/
        duplicate), or None (no match)."""
        npred = self._norm(predicted)
        targets = [g["gold_url"]] + ([g["original_url"]] if g.get("original_url") else [])
        # L1 URL / L2 entity -> exact
        if any(npred == self._norm(t) for t in targets):
            return "exact"
        pp = parse_url(predicted)
        if _entity_match(pp, g["repo"], g["entity_id"], g["entity_type"], self.renames):
            return "exact"
        # L3 alias-resolve both sides -> dup
        apred = self._alias(predicted)
        for t in targets:
            if apred == self._alias(t):
                return "dup"
        pa = parse_url(apred)
        if _entity_match(pa, g["repo"], g["entity_id"], g["entity_type"], self.renames):
            return "dup"
        return None

    def _match(self, predicted: str, g: dict) -> bool:
        return self._match_kind(predicted, g) is not None

    def is_correct(self, example_id: str, predicted_url: str) -> bool:
        g = self.gold.get(example_id)
        if g is None:
            raise KeyError(f"unknown example id: {example_id}")
        return self._match(predicted_url or "", g)

    def match(self, predicted_url: str, gold_record: dict) -> bool:
        """Score against an inline gold record (keys: gold_url, repo, entity_type,
        entity_id, optional original_url) — no gold file needed."""
        return self._match(predicted_url or "", gold_record)

    def match_kind(self, example_id: str, predicted_url: str) -> str | None:
        """'exact' | 'dup' | None — how (or whether) the prediction matched the gold."""
        g = self.gold.get(example_id)
        if g is None:
            raise KeyError(f"unknown example id: {example_id}")
        return self._match_kind(predicted_url or "", g)

    def score(self, predictions: dict) -> dict:
        """predictions: {example_id: predicted_url}. Returns accuracy + per-example."""
        per = {}
        for eid in self.gold:
            pred = predictions.get(eid)
            per[eid] = bool(pred) and self.is_correct(eid, pred)
        n = len(self.gold)
        correct = sum(per.values())
        return {"n": n, "correct": correct,
                "accuracy": correct / n if n else 0.0, "per_example": per}


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Score RepoRef predictions.")
    ap.add_argument("--gold", required=True)
    ap.add_argument("--predictions", required=True,
                    help="jsonl with {'id':..., 'predicted_url':...} per line")
    args = ap.parse_args()
    s = Scorer(args.gold)
    preds = {}
    for line in Path(args.predictions).read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            preds[r["id"]] = r.get("predicted_url") or r.get("predicted")
    rep = s.score(preds)
    print(f"accuracy: {100*rep['accuracy']:.2f}%  ({rep['correct']}/{rep['n']})")


if __name__ == "__main__":
    _main()
