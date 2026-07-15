"""Prepare or execute the frozen ArticleV1 canonical-window classifier study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustworthy_agent.article_v1.classification import execute, prepare_execution


def main() -> None:
    """Validate prerequisites, freeze the execution plan, or execute it.

    Side effects are limited to the declared ArticleV1 classifier output paths.
    Preparation performs no fitting; execution requires the pre-existing frozen
    plan and never invokes V6, evidence bundles, scenarios, the FSM, or safety.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--no-shadow-replay", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = (
        prepare_execution(root)
        if args.prepare_only
        else execute(root, shadow_replay=not args.no_shadow_replay)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
