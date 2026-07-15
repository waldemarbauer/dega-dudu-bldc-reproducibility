"""Git provenance helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_identity(repo_root: Path) -> dict[str, str]:
    """Return git commit and dirty-state provenance.

    Purpose:
        Attach source identity to audit and provenance manifests.
    Parameters:
        repo_root: Repository root.
    Return value:
        Mapping with commit SHA and dirty-state flag.
    Raised exceptions:
        None; unavailable git information is reported explicitly.
    Scientific assumptions:
        None.
    Side effects:
        Executes read-only git commands.
    Reproducibility implications:
        Source identity can be recorded without absolute developer paths.
    """

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"git_commit": "UNKNOWN", "git_dirty": "UNKNOWN", "git_error": str(exc)}
    return {"git_commit": commit, "git_dirty": str(bool(status))}
