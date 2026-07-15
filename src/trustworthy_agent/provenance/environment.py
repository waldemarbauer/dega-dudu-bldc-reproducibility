"""Environment provenance skeleton."""

from __future__ import annotations

import platform
import sys


def basic_environment() -> dict[str, str]:
    """Return basic interpreter and platform provenance.

    Purpose:
        Provide non-invasive environment metadata for later manifests.
    Parameters:
        None.
    Return value:
        Mapping with Python version, OS, and architecture.
    Raised exceptions:
        None.
    Scientific assumptions:
        None.
    Side effects:
        None.
    Reproducibility implications:
        Captures stable environment fields without running experiments.
    """

    return {
        "python_version": sys.version,
        "os": platform.system(),
        "architecture": platform.machine(),
    }
