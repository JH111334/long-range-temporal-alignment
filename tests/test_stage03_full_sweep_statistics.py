"""Unit tests for Stage 03 Full 54-Pair Causal Sweep Statistics, Delta_G metrics, and Hypotheses."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from temporal_alignment.causal_evaluation.clustered_mcnemar import _exact_binomial_p


def test_full_sweep_artifacts_exist():
    project_root = Path(__file__).resolve().parent.parent
    run_dir = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "runs"
        / "RUN_FULL_SWEEP_54P_20261019_161756"
    )

    required_files = [
        "PAIRED_STATISTICS.json",
        "GATES.json",
        "GOLD_AUDIT.json",
        "MECHANISM_TRACE.jsonl",
        "RAW_PREDICTIONS.jsonl",
        "RUN_CONFIG.json",
        "RUN_VERDICT.md",
        "TRACE_INDEX.json",
    ]

    for fname in required_files:
        fpath = run_dir / fname
        assert fpath.exists(), f"Missing required full sweep artifact: {fname}"

    otel_spans = run_dir / "traces" / "OTEL_SPANS.jsonl"
    assert otel_spans.exists(), "Missing OTEL_SPANS.jsonl"


def test_full_sweep_sample_budget_integrity():
    project_root = Path(__file__).resolve().parent.parent
    stats_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "runs"
        / "RUN_FULL_SWEEP_54P_20261019_161756"
        / "PAIRED_STATISTICS.json"
    )

    with open(stats_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    budget = data["sample_budget"]
    assert budget["total_pairs"] == 54
    assert budget["gold_flip_pairs"] == 36
    assert budget["gold_same_pairs"] == 18
    assert budget["total_questions"] == 108
    assert budget["unique_clusters"] == 23


def test_exact_binomial_mcnemar_and_table():
    # Contingency table for 108 questions:
    # a: both correct = 2
    # b: Base correct, Arm B incorrect = 59
    # c: Arm B correct, Base incorrect = 44
    # d: both incorrect = 3
    # Total = 108
    b = 59
    c = 44
    p_val = _exact_binomial_p(b, c)

    assert math.isclose(p_val, 0.167444, abs_tol=1e-4)
    assert p_val > 0.05, "Expected p > 0.05 (no significant difference over base)"


def test_mechanism_trace_delta_g_and_levels():
    project_root = Path(__file__).resolve().parent.parent
    trace_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "runs"
        / "RUN_FULL_SWEEP_54P_20261019_161756"
        / "MECHANISM_TRACE.jsonl"
    )

    traces = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    assert len(traces) == 54, f"Expected 54 traces, got {len(traces)}"

    flip_traces = [t for t in traces if t.get("gold_flips")]
    assert len(flip_traces) == 36

    gold_directed_count = sum(1 for t in flip_traces if t.get("is_gold_directed"))
    assert gold_directed_count == 13

    rate = gold_directed_count / len(flip_traces)
    assert math.isclose(rate, 0.3611, abs_tol=1e-3)

    mean_delta_g = sum(t["delta_g"] for t in flip_traces) / len(flip_traces)
    assert math.isclose(mean_delta_g, -0.058017, abs_tol=1e-3)

    for t in traces:
        assert t["d_z"] > 0.0, "D_z should be positive"
        assert t["d_w"] > 1e-5, "D_w should be positive"
        assert "s_seam_l14" in t
        assert "s_comp_l14" in t


def test_hypotheses_and_gates_verdicts():
    project_root = Path(__file__).resolve().parent.parent
    stats_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "runs"
        / "RUN_FULL_SWEEP_54P_20261019_161756"
        / "PAIRED_STATISTICS.json"
    )

    with open(stats_file, "r", encoding="utf-8") as f:
        stats = json.load(f)

    hypotheses = stats["hypotheses_verdict"]
    assert hypotheses["H1_representation_encoding"]["status"] == "REFUTED"
    assert hypotheses["H2_architecture_routing"]["status"] == "REFUTED"
    assert hypotheses["H3_decision_calibration_greedy_shield"]["status"] == "INCONCLUSIVE"
    assert hypotheses["H4_semantic_mapping_objectives"]["status"] == "SUPPORTED"

    assert stats["overall_gate_verdict"] == "NO_GO_SCALE"

    gates_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "runs"
        / "RUN_FULL_SWEEP_54P_20261019_161756"
        / "GATES.json"
    )

    with open(gates_file, "r", encoding="utf-8") as f:
        gates_data = json.load(f)

    gates = gates_data["gates"]
    assert "PASS (0 optimizer steps" in gates["P0_frozen_backbone"]
    assert "PASS" in gates["P1_delivery_dw"]
    assert "PASS" in gates["P2_sensitivity_logits"]
    assert "INCONCLUSIVE" in gates["P3_selectivity_margin"]
    assert "FAIL" in gates["P4_gold_direction_rate"]
    assert "FAIL" in gates["P5_semantic_both_correct"]
    assert "PASS" in gates["P6_negative_control_stability"]
    assert "FAIL" in gates["P7_statistical_gain_vs_base"]
    assert gates["P8_overall_causal_gate"] == "NO_GO_SCALE"


def test_writeback_ledger_has_full_sweep_entry():
    project_root = Path(__file__).resolve().parent.parent
    ledger_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "writebacks"
        / "WRITEBACK_LEDGER.jsonl"
    )

    entries = []
    with open(ledger_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    entry_ids = [e["entry_id"] for e in entries]
    assert "WB_20261019_002_FULL_SWEEP_54P" in entry_ids
