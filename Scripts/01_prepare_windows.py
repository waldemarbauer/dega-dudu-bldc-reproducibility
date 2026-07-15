"""Create canonical 0.8-s window provenance and the ArticleV1 feature corpus.

This bootstrap removes the private-project dependency on historical V2 window
metadata. It regenerates the exact 25 non-overlapping 40,000-sample windows per
acquisition from the pinned raw CSV files, using the same raw-window hash
algorithm as the ArticleV1 extractor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from trustworthy_agent.article_v1.pipeline import (
    RAW_HASH_DOMAIN_SEPARATOR,
    build_article_v1_window_analysis,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/experiments/article_v1/article_window_analysis_v1.yaml"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_window_hash(raw: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(RAW_HASH_DOMAIN_SEPARATOR)
    for values, dtype in (
        (raw[:, 0], "<i8"),
        (raw[:, 1], "<f8"),
        (raw[:, 2], "<f8"),
        (raw[:, 3], "<f8"),
    ):
        digest.update(np.asarray(values, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def read_raw(path: Path) -> np.ndarray:
    values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if values.shape != (1_000_000, 4) or not np.all(np.isfinite(values)):
        raise ValueError(f"RAW_VALUES_INVALID:{path.name}")
    sample = values[:, 0].astype(np.int64)
    if not np.array_equal(values[:, 0], sample) or not np.array_equal(
        sample, np.arange(1_000_000, dtype=np.int64)
    ):
        raise ValueError(f"RAW_SAMPLE_INDEX_INVALID:{path.name}")
    if not np.allclose(values[:, 1], sample / 50_000.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"RAW_TIME_SYNCHRONIZATION_FAILURE:{path.name}")
    return values


def main() -> int:
    protocol: dict[str, Any] = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    raw_root = ROOT / protocol["dataset"]["raw_source_root"]
    out = ROOT / "Data/AnalysisData/V2/Windows"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for source in protocol["raw_acquisitions"]:
        acquisition_id = str(source["acquisition_id"])
        source_path = raw_root / str(source["source_file"])
        actual = sha256_file(source_path)
        expected = str(source["source_sha256"])
        if actual != expected:
            raise ValueError(f"SOURCE_SHA256_MISMATCH:{acquisition_id}")
        raw = read_raw(source_path)
        for window_index in range(25):
            start = window_index * 40_000
            end = start + 40_000
            rows.append(
                {
                    "window_id": f"{acquisition_id}:{start}:{end}",
                    "source_group_id": acquisition_id,
                    "canonical_class": str(source["canonical_class"]),
                    "window_index_within_acquisition": window_index,
                    "start_sample": start,
                    "end_sample": end,
                    "sampling_frequency_hz": 50_000,
                    "samples_per_window": 40_000,
                    "raw_window_hash": raw_window_hash(raw[start:end]),
                    "source_acquisition_hash": expected,
                }
            )
    if len(rows) != 200:
        raise ValueError(f"EXPECTED_200_WINDOWS_GOT_{len(rows)}")
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, out / "window_provenance.parquet", compression="zstd")
    pq.write_table(
        table.select(
            [
                "window_id",
                "source_group_id",
                "canonical_class",
                "window_index_within_acquisition",
                "start_sample",
                "end_sample",
            ]
        ),
        out / "window_index.parquet",
        compression="zstd",
    )
    result = build_article_v1_window_analysis(ROOT)
    result["canonical_window_bootstrap"] = {
        "window_count": 200,
        "provenance_path": "Data/AnalysisData/V2/Windows/window_provenance.parquet",
        "provenance_sha256": sha256_file(out / "window_provenance.parquet"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
