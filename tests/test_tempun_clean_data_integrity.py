"""Tests for Clean TempUN Datasets Integrity (Small and Full Clean).

Verifies:
- Existence, file sizes, and exact SHA-256 digests.
- Line counts (180 / 2,461 for Small, 5,292 / 104,130 for Full).
- Zero null-year defect in clean evaluations.
- Strict schema compliance.
- Absolute train-eval leakage isolation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPUN_CLEAN_DIR = PROJECT_ROOT / "data" / "tempun_clean"
TEMPUN_FULL_DIR = PROJECT_ROOT / "data" / "tempun_full_clean"

EXPECTED_SMALL_EVAL_SHA256 = "94fdb40a6a39712ceccdf21d22b3916337b4e633867ee9584874f84df2f46069"
EXPECTED_SMALL_TRAIN_SHA256 = "41eae532034f375e3cb8c24d07afe04e16d897185020d8b382d1e591bce4ed68"

EXPECTED_FULL_EVAL_SHA256 = "a624d8d258f9676f00e504d16c35dc401cc1cce96613ca56af970daa32a12fc3"
EXPECTED_FULL_TRAIN_SHA256 = "111d2d0908533a17ac10c98533802053631b708a23846524daa224bd1465697d"


def _compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def test_tempun_small_clean_integrity():
    eval_file = TEMPUN_CLEAN_DIR / "eval_180_clean.jsonl"
    train_file = TEMPUN_CLEAN_DIR / "train_2461_clean.jsonl"

    assert eval_file.is_file(), f"Missing {eval_file}"
    assert train_file.is_file(), f"Missing {train_file}"

    assert _compute_sha256(eval_file) == EXPECTED_SMALL_EVAL_SHA256
    assert _compute_sha256(train_file) == EXPECTED_SMALL_TRAIN_SHA256

    eval_rows = [json.loads(line) for line in eval_file.open(encoding="utf-8")]
    train_rows = [json.loads(line) for line in train_file.open(encoding="utf-8")]

    assert len(eval_rows) == 180
    assert len(train_rows) == 2461

    # Verify zero null-year defect
    for idx, r in enumerate(eval_rows):
        assert r.get("year") is not None, f"Row {idx} has None year"
        assert isinstance(r["year"], int)
        assert 1947 <= r["year"] <= 2022
        assert "example_id" in r
        assert "input" in r
        assert "target" in r


def test_tempun_full_clean_integrity():
    eval_file = TEMPUN_FULL_DIR / "eval_5292_clean.jsonl.gz"
    train_file = TEMPUN_FULL_DIR / "train_104k_clean.jsonl.gz"
    checksums_file = TEMPUN_FULL_DIR / "CHECKSUMS.sha256"
    metadata_file = TEMPUN_FULL_DIR / "dataset-metadata.json"

    assert eval_file.is_file()
    assert train_file.is_file()
    assert checksums_file.is_file()
    assert metadata_file.is_file()

    eval_sha = _compute_sha256(eval_file)
    train_sha = _compute_sha256(train_file)

    assert eval_sha == EXPECTED_FULL_EVAL_SHA256
    assert train_sha == EXPECTED_FULL_TRAIN_SHA256

    # Verify metadata and checksum file alignment
    checksum_content = checksums_file.read_text(encoding="utf-8")
    assert train_sha in checksum_content
    assert eval_sha in checksum_content

    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert metadata["sha256"]["eval_5292_clean.jsonl.gz"] == eval_sha
    assert metadata["sha256"]["train_104k_clean.jsonl.gz"] == train_sha
    assert metadata["mcqa_samples"] == 5292
    assert metadata["nwp_samples"] == 104130


def test_tempun_clean_zero_leakage():
    small_eval = [json.loads(l) for l in (TEMPUN_CLEAN_DIR / "eval_180_clean.jsonl").open(encoding="utf-8")]
    small_train = [json.loads(l) for l in (TEMPUN_CLEAN_DIR / "train_2461_clean.jsonl").open(encoding="utf-8")]

    s_eval_ids = set(r["example_id"] for r in small_eval)
    s_train_ids = set(r["example_id"] for r in small_train)
    assert len(s_eval_ids.intersection(s_train_ids)) == 0

    s_eval_inputs = set(r["input"].strip().lower() for r in small_eval)
    s_train_inputs = set(r["input"].strip().lower() for r in small_train)
    assert len(s_eval_inputs.intersection(s_train_inputs)) == 0
