"""Tests for Stage 03 Representation-First Canonical Evaluation, Clustered Statistics, and Counterfactuals."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from temporal_alignment.causal_evaluation.canonical_evaluator import (
    CanonicalEvaluationConfig,
    RawPredictionRecord,
    generate_scoreboard_from_predictions,
)
from temporal_alignment.causal_evaluation.clustered_mcnemar import (
    _exact_binomial_p,
    compute_clustered_bootstrap_ci,
    compute_clustered_mcnemar,
)


def test_scoreboard_computation_hygiene():
    """Test scoreboard generation directly from raw predictions with label balance gate."""
    mock_predictions = [
        {
            "run_id": "test_run",
            "example_id": f"ex_{i}",
            "source_record_id": f"src_{i % 3}",
            "event_id": f"ev_{i % 2}",
            "temporal_cluster_id": "clust_1",
            "checkpoint_sha256": "mock_sha",
            "tensor_fingerprint": "mock_fp",
            "condition": "A0_BASE",
            "year_true": 1950,
            "year_used": 1950,
            "treatment_status": "ELIGIBLE_SINGLE_YEAR",
            "gold_semantic_answer": "Option A" if i % 2 == 0 else "Option B",
            "gold_display_label": "A" if i % 2 == 0 else "B",
            "raw_generation": "(A)",
            "parsed_prediction": "A" if i < 8 else "B",
            "is_valid": True,
            "is_correct": (i % 2 == 0 and i < 8) or (i % 2 == 1 and i >= 8),
            "seed": 42,
            "prompt_template_sha": "t_sha",
            "parser_sha": "p_sha",
            "generation_config_sha": "g_sha",
        }
        for i in range(10)
    ]

    sb = generate_scoreboard_from_predictions(mock_predictions)
    assert sb["summary"]["total_count"] == 10
    assert sb["summary"]["valid_count"] == 10
    assert sb["summary"]["invalid_count"] == 0
    assert sb["summary"]["invalid_rate"] == 0.0
    assert sb["hygiene_gate"]["max_label_frequency"] == 0.8
    assert sb["hygiene_gate"]["is_label_collapsed"] is True  # 80% > 50%
    assert sb["hygiene_gate"]["hygiene_passed"] is False


def test_clustered_mcnemar_and_bootstrap():
    """Test clustered McNemar calculation and bootstrap confidence intervals."""
    # Build 20 synthetic paired predictions across 4 clusters
    preds_m1 = []
    preds_m2 = []
    for i in range(20):
        c_id = f"cluster_{i // 5}"
        ex_id = f"item_{i}"
        # Model 1 correct 14/20, Model 2 correct 10/20
        corr1 = (i < 14)
        corr2 = (i < 10)
        preds_m1.append({"example_id": ex_id, "is_correct": corr1, "source_record_id": c_id})
        preds_m2.append({"example_id": ex_id, "is_correct": corr2, "source_record_id": c_id})

    res = compute_clustered_mcnemar(preds_m1, preds_m2, cluster_field="source_record_id")
    assert res.contingency_table["total"] == 20
    assert res.contingency_table["a"] == 10  # both correct (0..9)
    assert res.contingency_table["b"] == 4   # M1 correct, M2 wrong (10..13)
    assert res.contingency_table["c"] == 0   # M1 wrong, M2 correct
    assert res.contingency_table["d"] == 6   # both wrong (14..19)
    assert res.delta_accuracy == pytest.approx(0.20, abs=1e-4)
    assert res.cluster_count == 4
    assert res.p_value_exact is not None
    assert isinstance(res.bootstrap_ci_95, tuple)
    assert len(res.bootstrap_ci_95) == 2


def test_exact_binomial_p_values():
    """Verify exact binomial p-values on edge cases."""
    # When b = 0, c = 0, p = 1.0
    assert _exact_binomial_p(0, 0) == 1.0
    # When b = 4, c = 0, p = 2 * (0.5)^4 = 0.125
    assert _exact_binomial_p(4, 0) == pytest.approx(0.125, abs=1e-5)


def test_grounded_counterfactual_manifest():
    """Verify grounded counterfactual datasets exist, match specs, and have valid JSONL."""
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "data/counterfactual/MANIFEST.json"
    assert manifest_path.is_file(), "Counterfactual MANIFEST.json must exist"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["coverage"]["gold_flip_pairs"] == 36
    assert manifest["coverage"]["gold_same_pairs"] == 18

    flip_file = repo_root / "data/counterfactual/grounded_gold_flip_pairs.jsonl"
    assert flip_file.is_file()
    with open(flip_file, "r", encoding="utf-8") as f:
        flip_lines = [json.loads(l) for l in f if l.strip()]
    assert len(flip_lines) == 72  # 36 pairs * 2

    # Check pair matching and gold flipping
    by_pair = {}
    for r in flip_lines:
        by_pair.setdefault(r["pair_id"], []).append(r)
    assert len(by_pair) == 36
    for pid, items in by_pair.items():
        assert len(items) == 2
        # Target must flip
        assert items[0]["target"].lower() != items[1]["target"].lower()
        # Years must differ
        assert items[0]["year"] != items[1]["year"]
        # Entity and relation must be identical
        assert items[0]["entity"] == items[1]["entity"]
        assert items[0]["relation"] == items[1]["relation"]
        assert items[0]["treatment_status"] == "ELIGIBLE_GOLD_FLIP"


def test_temporal_mechanism_tracer_math():
    """Verify mathematical properties of temporal difference metrics."""
    from temporal_alignment.causal_evaluation.mechanism_tracer import TemporalMechanismTracer
    from temporal_alignment.models.fone_canchi_40d import compute_fixed_fone_canchi_40d, extract_canchi_angles_from_40d

    # Test Dz: identical years have Dz = 0
    v1 = compute_fixed_fone_canchi_40d(1980)
    v2 = compute_fixed_fone_canchi_40d(1980)
    z1 = torch.tensor(v1, dtype=torch.float32)
    z2 = torch.tensor(v2, dtype=torch.float32)
    assert (z1 - z2).norm(p=2).item() == 0.0

    # Test Dz: different years have positive distance Dz > 0
    v3 = compute_fixed_fone_canchi_40d(2020)
    z3 = torch.tensor(v3, dtype=torch.float32)
    d_z = (z1 - z3).norm(p=2).item()
    assert d_z > 1.0

    # Test angle extraction
    th1, zc1 = extract_canchi_angles_from_40d(v1)
    th3, zc3 = extract_canchi_angles_from_40d(v3)
    assert -math.pi <= th1 <= math.pi
    assert -math.pi <= th3 <= math.pi
    assert zc3 > zc1  # 2020 has larger radius coordinate than 1980

