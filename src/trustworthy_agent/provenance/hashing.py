"""General deterministic hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def md5_file(path: Path) -> str:
    """Compute an MD5 digest for a file.

    Purpose:
        Verify the externally specified DUDU-BLDC archive checksum.
    Parameters:
        path: File path to hash.
    Return value:
        Hex MD5 digest.
    Raised exceptions:
        OSError for file-read failures.
    Scientific assumptions:
        MD5 is used only because the canonical dataset specification supplies
        an MD5 checksum; SHA-256 is also computed for provenance.
    Side effects:
        Reads the file only.
    Reproducibility implications:
        Enforces the pinned archive identity.
    """

    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    """Compute a SHA-256 digest for a file.

    Purpose:
        Support future dataset and artifact manifests.
    Parameters:
        path: File path to hash.
    Return value:
        Hex SHA-256 digest.
    Raised exceptions:
        OSError for file-read failures.
    Scientific assumptions:
        None.
    Side effects:
        Reads the file only.
    Reproducibility implications:
        Provides stable artifact identity.
    """

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
