"""Generate the frozen ArticleV1 canonical-window prerequisite artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from trustworthy_agent.article_v1.pipeline import build_article_v1_window_analysis


def main() -> int:
    """Run the ArticleV1 prerequisite generator from the repository root.

    Returns
    -------
    int
        Zero after all declared artifacts pass generation-time validation.

    Raises
    ------
    OSError, ValueError
        Propagated fail-closed when an input, identity, feature, or assignment
        contract is invalid.

    Side Effects
    ------------
    Writes only the declared ArticleV1 AnalysisData and Output artifacts.
    """

    root = Path(__file__).resolve().parents[2]
    result = build_article_v1_window_analysis(root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
