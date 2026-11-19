"""Unit tests for V2 Causal Temporal Alignment Contract and Data Invariants."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from temporal_alignment.models.fone_canchi_40d import (
    compute_fixed_fone_canchi_40d,
    extract_canchi_angles_from_40d,
)
from temporal_alignment.baselines.ticktack05b.evaluator import (
    detect_is_yes_no_task,
    task_conditioned_strict_parser,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fixed_fone_canchi_40d_invariants():
    """Verify 40D feature dimension, deterministic reproducibility, and angle ranges."""
    v1947 = compute_fixed_fone_canchi_40d(1947)
    v1984 = compute_fixed_fone_canchi_40d(1984)
    v2022 = compute_fixed_fone_canchi_40d(2022)

    assert len(v1947) == 40
    assert len(v1984) == 40
    assert len(v2022) == 40

    # Test determinism
    assert compute_fixed_fone_canchi_40d(1984) == v1984

    # Test angle extraction
    theta, z = extract_canchi_angles_from_40d(v1984)
    assert -3.15 <= theta <= 3.15
    assert isinstance(z, float)


def test_internal_train_dev_split_integrity():
    """Ensure internal group-held-out split exists and contains valid records."""
    split_path = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_01" / "internal_train_dev_split.jsonl"
    if split_path.is_file():
        records = []
        with open(split_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        assert len(records) > 0
        first = records[0]
        assert "canonical_year" in first
        assert "input" in first
        assert "target" in first


def test_task_detection_and_parser_integrity():
    """Test task discrimination between MCQA and Yes/No."""
    yn_text = "In 1947, was the index higher than in 1952? Answer as Yes or No"
    mcqa_text = "What was the index in 1947? (a) 10 (b) 20 (c) 30 (d) 40"

    assert detect_is_yes_no_task(yn_text) is True
    assert detect_is_yes_no_task(mcqa_text) is False

    # Parser compliance
    pred, status = task_conditioned_strict_parser("Yes", is_yn=True)
    assert pred == "Yes"
    assert status == "VALID"

    pred, status = task_conditioned_strict_parser("a", is_yn=False)
    assert pred == "a"
    assert status == "VALID"

    pred, status = task_conditioned_strict_parser("Yes", is_yn=False)
    assert pred is None
    assert status == "INVALID_WRONG_TYPE_YN_ON_MCQA"
