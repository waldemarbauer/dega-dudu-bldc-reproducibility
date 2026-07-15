"""One-command ArticleV1 reproduction orchestrator.

Stages are intentionally explicit and fail closed. The external dataset is
fetched only when requested or when absent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    (0, "fetch dataset", "scripts/00_fetch_dataset.py"),
    (1, "prepare canonical windows and 28 features", "scripts/01_prepare_windows.py"),
    (2, "train 48 classifiers and infer held-out windows", "scripts/02_train_classifiers.py"),
    (3, "generate real evidence and 768 bundles", "scripts/03_generate_evidence.py"),
    (4, "run minimal XAI", "scripts/04_run_xai.py"),
    (5, "run 12 representative DEGA executions", "scripts/05_run_dega.py"),
    (6, "generate manuscript figures", "scripts/06_generate_figures.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-stage", type=int, default=0, choices=range(7))
    parser.add_argument("--to-stage", type=int, default=6, choices=range(7))
    parser.add_argument("--skip-fetch", action="store_true")
    args = parser.parse_args()
    if args.from_stage > args.to_stage:
        parser.error("--from-stage must not exceed --to-stage")
    for number, label, script in STAGES:
        if number < args.from_stage or number > args.to_stage:
            continue
        if number == 0 and args.skip_fetch:
            print("[skip] stage 0: dataset fetch")
            continue
        print(f"[run] stage {number}: {label}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    print("DEGA ARTICLEV1 REPRODUCTION PIPELINE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
