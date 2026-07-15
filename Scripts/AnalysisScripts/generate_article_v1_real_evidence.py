"""Generate and replay the frozen ArticleV1 E1.4 diagnostic evidence layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustworthy_agent.article_v1.real_evidence import execute, prepare_execution, replay


def main() -> int:
    """Run preflight, scientific generation, and optional complete replay."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    prepare_execution(root)
    summary = execute(root)
    if args.replay:
        summary["replay"] = replay(root)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
