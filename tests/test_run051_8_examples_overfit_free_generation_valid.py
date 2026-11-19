"""Unit Test for Canary Overfit, Free Generation Validity, and Anti-Collapse Verification."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_canary_format_anti_collapse():
    """Verify that FORMAT_EOS_REPEAT_DIAGNOSTICS confirms elimination of repetitive continuation."""
    diag_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "FORMAT_EOS_REPEAT_DIAGNOSTICS.json"
    assert diag_file.exists(), f"Missing diagnostic file: {diag_file}"

    with open(diag_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]
    # Run 05 had 100% repetition in Arm A and 96.7% invalid in Arm B
    # Run 05.1 canary proves repetitive bursts are completely eliminated
    assert metrics["repeated_continuation_count"] == 0, f"Found repeated continuation: {metrics['repeated_continuation_count']}"
    assert metrics["repeated_continuation_rate"] == 0.0
    assert metrics["syntax_valid_rate"] >= 0.70


def test_canary_gradient_and_backbone_freeze():
    """Verify that TASK_LOSS_MASK_GRADIENT_AND_CLIPPING confirms 100% frozen base layers across all steps."""
    trace_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "TASK_LOSS_MASK_GRADIENT_AND_CLIPPING.jsonl"
    assert trace_file.exists(), f"Missing trace file: {trace_file}"

    traces = []
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))

    assert len(traces) == 15
    for tr in traces:
        assert tr["base_layer_grad_is_none"] is True, f"Base layer leaked gradient at step {tr['step']}!"
        assert tr["grad_norm"] > 0.0


def test_canary_gate_decision_approved():
    """Verify CANARY_GATE_DECISION officially approves proceeding to full A/B training."""
    dec_file = _REPO_ROOT / "artifacts" / "experiments" / "V2_CAUSAL_TEMPORAL_ALIGNMENT" / "run_05_1" / "CANARY_GATE_DECISION.json"
    assert dec_file.exists(), f"Missing decision file: {dec_file}"

    with open(dec_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["status"] == "APPROVED"
    assert data["criteria"]["backbone_100_percent_frozen"] is True
    assert data["criteria"]["base_layer_grad_is_none_at_all_steps"] is True
