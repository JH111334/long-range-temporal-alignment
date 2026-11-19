"""Regression tests for clean verified MCQA4 dataset in V2 Run 04."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_clean_dataset_single_truth_contract():
    """Verify that 100% of items in TRAIN and DEV have exactly one true option."""
    run04_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    train_clean = run04_dir / "MCQA4_TRAIN_CLEAN.jsonl"
    dev_clean = run04_dir / "MCQA4_DEV_CLEAN.jsonl"

    assert train_clean.is_file(), "Missing MCQA4_TRAIN_CLEAN.jsonl"
    assert dev_clean.is_file(), "Missing MCQA4_DEV_CLEAN.jsonl"

    for file_path, expected_len in [(train_clean, 400), (dev_clean, 120)]:
        with open(file_path, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]

        assert len(items) == expected_len

        for item in items:
            v_map = {item["year_eval"]: item["val_eval"], item["year_base"]: item["val_base"]}
            opts_ast = item["options_ast"]
            gold = item["target"]

            true_letters = []
            for ltr, ast in opts_ast.items():
                y_l, op, y_r = ast["year_left"], ast["operator"], ast["year_right"]
                vl, vr = v_map[y_l], v_map[y_r]
                is_true = (vl > vr) if op == "gt" else ((vl < vr) if op == "lt" else abs(vl - vr) < 1e-6)
                if is_true:
                    true_letters.append(ltr)

            assert len(true_letters) == 1, f"Item {item['sample_id']} has {len(true_letters)} true options: {true_letters}"
            assert true_letters[0] == gold, f"Item {item['sample_id']} true letter {true_letters[0]} != gold {gold}"


def test_clean_dataset_direction_and_letter_balance():
    """Verify 50/50 direction balance and uniform 25% letter balance."""
    run04_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    dev_clean = run04_dir / "MCQA4_DEV_CLEAN.jsonl"

    with open(dev_clean, "r", encoding="utf-8") as f:
        dev_items = [json.loads(line) for line in f if line.strip()]

    dirs = [item["direction"] for item in dev_items]
    letters = [item["target"] for item in dev_items]

    assert dirs.count("higher") == 60
    assert dirs.count("lower") == 60

    assert letters.count("a") == 30
    assert letters.count("b") == 30
    assert letters.count("c") == 30
    assert letters.count("d") == 30


def test_clean_dataset_zero_leakage():
    """Verify group-disjoint entities and zero overlap with eval_mcqa.jsonl."""
    run04_dir = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_04"
    train_clean = run04_dir / "MCQA4_TRAIN_CLEAN.jsonl"
    dev_clean = run04_dir / "MCQA4_DEV_CLEAN.jsonl"
    eval_benchmark = _REPO_ROOT / "data" / "tempun" / "eval_mcqa.jsonl"

    with open(train_clean, "r", encoding="utf-8") as f:
        train_entities = {json.loads(line)["entity"] for line in f if line.strip()}
    with open(dev_clean, "r", encoding="utf-8") as f:
        dev_entities = {json.loads(line)["entity"] for line in f if line.strip()}
    with open(eval_benchmark, "r", encoding="utf-8") as f:
        eval_inputs = {json.loads(line).get("input", "") for line in f if line.strip()}

    assert len(train_entities.intersection(dev_entities)) == 0
    assert train_entities == {"GBXE", "DVPR", "RRHM"}
    assert dev_entities == {"BCXZ", "GVRO"}

    with open(train_clean, "r", encoding="utf-8") as f:
        train_questions = {json.loads(line)["question"] for line in f if line.strip()}
    assert len(train_questions.intersection(eval_inputs)) == 0
