"""Unit tests for V2 Run 03 multi-task dataset, schema, and evaluators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_v2_run03_dataset_group_disjointness():
    """Verify train and dev entities are strictly group-disjoint."""
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    train_file = run03_dir / "MCQA4_TRAIN.jsonl"
    dev_file = run03_dir / "MCQA4_DEV.jsonl"

    assert train_file.is_file(), f"Missing train file: {train_file}"
    assert dev_file.is_file(), f"Missing dev file: {dev_file}"

    with open(train_file, "r", encoding="utf-8") as f:
        train_entities = {json.loads(line)["entity"] for line in f if line.strip()}

    with open(dev_file, "r", encoding="utf-8") as f:
        dev_entities = {json.loads(line)["entity"] for line in f if line.strip()}

    overlap = train_entities.intersection(dev_entities)
    assert len(overlap) == 0, f"Critical entity leakage detected between train and dev: {overlap}"
    assert train_entities == {"GBXE", "DVPR", "RRHM"}
    assert dev_entities == {"BCXZ", "GVRO"}


def test_v2_run03_dev_option_balance():
    """Verify dev choices are strictly balanced across a, b, c, d (25% each)."""
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    dev_file = run03_dir / "MCQA4_DEV.jsonl"

    with open(dev_file, "r", encoding="utf-8") as f:
        dev_targets = [json.loads(line)["target"].lower() for line in f if line.strip()]

    assert len(dev_targets) == 120
    counts = {k: dev_targets.count(k) for k in ["a", "b", "c", "d"]}
    assert counts == {"a": 30, "b": 30, "c": 30, "d": 30}, f"Imbalanced dev options: {counts}"


def test_v2_run03_hard_negatives_presence():
    """Verify dev set includes decreasing hard negatives."""
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    dev_file = run03_dir / "MCQA4_DEV.jsonl"

    with open(dev_file, "r", encoding="utf-8") as f:
        hard_negs = [json.loads(line) for line in f if not json.loads(line).get("is_increasing", True)]

    assert len(hard_negs) == 20
    assert len(hard_negs) / 120 == pytest.approx(0.1667, abs=1e-3)


def test_v2_run03_parser_compliance():
    """Verify task_conditioned_strict_parser correctly distinguishes MCQA vs Yes/No."""
    from temporal_alignment.baselines.ticktack05b.evaluator import (
        task_conditioned_strict_parser,
    )

    # Valid MCQA options
    pred, status = task_conditioned_strict_parser(" (c)", is_yn=False)
    assert pred == "c" and status == "VALID"

    pred, status = task_conditioned_strict_parser("The answer is (a).", is_yn=False)
    assert pred == "a" and status == "VALID"

    # Invalid MCQA: model returns Yes/No
    pred, status = task_conditioned_strict_parser("Yes", is_yn=False)
    assert pred is None and status == "INVALID_WRONG_TYPE_YN_ON_MCQA"

    # Valid Yes/No
    pred, status = task_conditioned_strict_parser(" Yes", is_yn=True)
    assert pred == "Yes" and status == "VALID"

    # Invalid Yes/No: model returns option letter
    pred, status = task_conditioned_strict_parser(" (b)", is_yn=True)
    assert pred is None and status == "INVALID_WRONG_TYPE_LETTER_ON_YN"


def test_v2_run03_preregistration_exists():
    """Verify pre-registration sealed document exists and contains required arms."""
    run03_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_03"
    prereg_file = run03_dir / "PRE_REGISTERED_RUN03.json"

    assert prereg_file.is_file(), "Missing PRE_REGISTERED_RUN03.json"
    with open(prereg_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "R3_A_RELATION_ONLY" in data["arms"]
    assert "R3_B_MCQA_ONLY" in data["arms"]
    assert "R3_C_JOINT_TRUE" in data["arms"]
    assert "R3_D_JOINT_SHUFFLED" in data["arms"]
    assert data["starting_checkpoint"]["sha256"] == "09acf723ea97401924a070b14775d09584d0a7a565c68d83c71218e814f72cb9"
