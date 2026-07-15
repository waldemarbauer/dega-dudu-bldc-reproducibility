"""Fetch and verify the pinned DUDU-BLDC dataset from Zenodo."""

from __future__ import annotations

import json
from pathlib import Path

from trustworthy_agent.data.downloader import fetch_dudu_bldc


if __name__ == "__main__":
    manifest = fetch_dudu_bldc(Path.cwd())
    print(json.dumps(manifest.to_json_dict(), indent=2, sort_keys=True))
