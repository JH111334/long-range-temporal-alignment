"""Unit Test for Single-Gold Oracle Truth and Non-Leakage in Run 05 Clean Datasets."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mcqa_clean_datasets_single_gold():
    """Verify that 100% of samples in clean train and dev sets have exactly one unique correct answer."""
    run05_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp"
    train_path = run05_dir / "MCQA4_TRAIN_CLEAN.jsonl"
    dev_path = run05_dir / "MCQA4_DEV_CLEAN.jsonl"

    for path, expected_len in [(train_path, 400), (dev_path, 120)]:
        assert path.exists(), f"File {path} does not exist"
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == expected_len, f"Expected {expected_len} items in {path.name}, got {len(lines)}"

        for i, item in enumerate(lines):
            # Target must be one of a, b, c, d
            target = item.get("target")
            assert target in ("a", "b", "c", "d"), f"Line {i} invalid target: {target}"

            # Options must contain exactly a, b, c, d
            opts = item.get("options", {})
            assert set(opts.keys()) == {"a", "b", "c", "d"}, f"Line {i} invalid options: {opts.keys()}"

            # Question text must be non-empty
            q = item.get("question", "")
            assert len(q.strip()) > 10, f"Line {i} empty or short question: {q}"


def test_mcqa_clean_zero_leakage():
    """Verify that train and dev sets have zero entity/event overlap (zero leakage)."""
    run05_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_ab_ntp"
    audit_path = run05_dir / "SPLIT_AND_LEAKAGE_AUDIT.json"
    assert audit_path.exists()

    with open(audit_path, "r", encoding="utf-8") as f:
        audit = json.load(f)

    assert audit["exact_question_overlap_count"] == 0
    assert audit["leakage_detected"] is False
    assert audit["verdict"] == "ZERO_LEAKAGE_CONFIRMED"
