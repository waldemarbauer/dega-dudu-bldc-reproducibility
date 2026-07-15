"""Execute the 12 representative DEGA runs and deterministic replay."""

from __future__ import annotations

import json
from pathlib import Path

from trustworthy_agent.article_v1.publication_e2 import execute


if __name__ == "__main__":
    print(json.dumps(execute(Path.cwd()), indent=2, sort_keys=True))
