"""Reproduce RepoRef results: run one model over a split with the real ReAct harness.

    export GITHUB_TOKEN=...             # read-only PAT for the GitHub tools
    export GEMINI_API_KEY=...           # provider key for the chosen model (see config/models.yaml)
    python -m reporef.run --model gemini-3-flash --split main --budget 10

Writes predictions.jsonl and prints accuracy (via reporef.scorer).
Per-run traces are written under runs/<model>/budget_<b>/.
"""
from __future__ import annotations
import argparse, json, logging, os, sys
from pathlib import Path

class _DropGeminiTempWarning(logging.Filter):
    def filter(self, record):
        return "planned for removal" not in record.getMessage()
logging.getLogger("LiteLLM").addFilter(_DropGeminiTempWarning())

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent / "config"
CACHE_DIR = Path(__file__).resolve().parent / "cache"
SPLITS = {
    "main":     ("data/main/instances.jsonl",            "data/gold/main.jsonl"),
    "extended": ("data/extras/extended/instances.jsonl", "data/extras/extended/gold.jsonl"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="a key under `models:` in config/models.yaml")
    ap.add_argument("--split", choices=SPLITS, default="main")
    ap.add_argument("--budget", type=int, default=10, help="max tool calls")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="predictions.jsonl")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    import yaml
    from .runners.single_run import run_single
    from .data.loader import load_examples
    from .scorer import Scorer

    models = yaml.safe_load((CONFIG_DIR / "models.yaml").read_text())["models"]
    if args.model not in models:
        sys.exit(f"unknown model '{args.model}'. available: {', '.join(models)}")
    model_cfg = models[args.model]

    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        print("warning: GITHUB_TOKEN not set — tool calls will only work from cache",
              file=sys.stderr)

    inst_path, gold_path = (ROOT / p for p in SPLITS[args.split])
    examples = load_examples(inst_path, gold_path)
    if args.limit:
        examples = examples[:args.limit]

    runs_dir = Path(args.runs_dir) / args.model / f"budget_{args.budget}"
    runs_dir.mkdir(parents=True, exist_ok=True)

    s = Scorer(gold_path)
    preds = {}
    with open(args.out, "w") as fo:
        for i, ex in enumerate(examples, 1):
            try:
                r = run_single(ex, args.model, model_cfg, args.budget,
                               CONFIG_DIR, token, cache_dir=CACHE_DIR)
                pred = r.evaluation.predicted or ""
                (runs_dir / f"{ex.id}.json").write_text(r.model_dump_json(indent=1))
            except Exception as e:
                pred = ""
                print(f"  [{ex.id}] run error: {e}", file=sys.stderr)
            preds[ex.id] = pred
            fo.write(json.dumps({"id": ex.id, "predicted_url": pred}) + "\n")
            kind = s.match_kind(ex.id, pred) if pred else None
            tag = "ok" if kind == "exact" else "ok, dup" if kind == "dup" else "x"
            print(f"[{i}/{len(examples)}] {ex.id} -> {pred}  ({tag})")

    kinds = [s.match_kind(e, u) if u else None for e, u in preds.items()]
    correct = sum(1 for k in kinds if k is not None)
    dup = sum(1 for k in kinds if k == "dup")
    n = len(preds)
    extra = f", {dup} via duplicate" if dup else ""
    print(f"\n{args.model}  {args.split}  budget={args.budget}: "
          f"{100*correct/n:.2f}%  ({correct}/{n}{extra})")


if __name__ == "__main__":
    main()
