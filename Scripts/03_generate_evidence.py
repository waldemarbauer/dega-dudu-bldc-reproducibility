"""Generate training-only references, temporal evidence and 768 EvidenceBundles."""

from __future__ import annotations

from _stage_runner import run_script


if __name__ == "__main__":
    run_script("Scripts/AnalysisScripts/generate_article_v1_real_evidence.py")
