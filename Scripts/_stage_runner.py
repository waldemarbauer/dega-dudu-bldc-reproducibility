"""Shared subprocess helper for publication-reproduction stage wrappers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(relative_path: str, *args: str) -> None:
    command = [sys.executable, str(ROOT / relative_path), *args]
    subprocess.run(command, cwd=ROOT, check=True)
