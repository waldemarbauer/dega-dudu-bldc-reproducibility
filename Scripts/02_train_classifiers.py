"""Train 48 ArticleV1 classifiers and generate held-out predictions."""

from __future__ import annotations

import argparse

from _stage_runner import run_script


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-shadow-replay", action="store_true")
    args = parser.parse_args()
    run_script("Scripts/AnalysisScripts/run_article_v1_window_classifiers.py", "--prepare-only")
    if not args.prepare_only:
        extra = ("--no-shadow-replay",) if args.no_shadow_replay else ()
        run_script("Scripts/AnalysisScripts/run_article_v1_window_classifiers.py", "--execute", *extra)
