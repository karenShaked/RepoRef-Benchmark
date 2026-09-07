#!/usr/bin/env python3
"""Convenience entry point — run from the repo root:

    python run.py --model gemini-3-flash --split main --budget 10 --limit 2

Wires the repo root (for `benchmark.*`) and `baseline-agents/` (for `reporef.*`)
onto sys.path, then delegates to the baseline ReAct runner.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                       # import benchmark.*
sys.path.insert(0, str(ROOT / "baseline-agents"))   # import reporef.*

from reporef.run import main

if __name__ == "__main__":
    main()
