"""Unit tests for Stage 03 Branch-Safe Writeback, Schema Compliance, and Ledger Invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_bottleneck_partition_files_exist():
    project_root = Path(__file__).resolve().parent.parent
    base_dir = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
    )

    required_files = [
        base_dir / "README.md",
        base_dir / "INDEX.md",
        base_dir / "config" / "bottleneck_experiment.yaml",
        base_dir / "config" / "otel_phoenix.yaml",
        base_dir / "config" / ".env.example",
        base_dir / "config" / "representation_interface.schema.json",
        base_dir / "config" / "thresholds.yaml",
        base_dir / "contracts" / "BRANCH_HANDOFF.md",
        base_dir / "contracts" / "REPRESENTATION_COMPATIBILITY.md",
        base_dir / "protocols" / "TRACING_METRIC_DEFINITIONS.md",
        base_dir / "protocols" / "CAUSAL_EVALUATION_CONTRACT.md",
        base_dir / "protocols" / "COUNTERFACTUAL_GOLD_AUDIT.md",
        base_dir / "writebacks" / "WRITEBACK_RULES.md",
        base_dir / "writebacks" / "WRITEBACK_LEDGER.jsonl",
    ]

    for f in required_files:
        assert f.exists(), f"Missing required partition file: {f}"


def test_representation_interface_schema_valid():
    project_root = Path(__file__).resolve().parent.parent
    schema_file = (
        project_root
        / "Reports"
        / "03_TARGETED_OBJECTIVES_MCQA"
        / "REPRESENTATION_BOTTLENECK"
        / "config"
        / "representation_interface.schema.json"
    )

    with open(schema_file, "r", encoding="utf-8") as f:
        schema = json.load(f)

    assert schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
    assert "properties" in schema
    assert "required" in schema
    assert "dimension" in schema["properties"]
    assert "granularity" in schema["properties"]


def test_writeback_ledger_invariants():
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

    assert len(entries) >= 1, "Ledger must contain at least the initialization entry"

    required_fields = [
        "entry_id",
        "timestamp",
        "transaction_type",
        "operator_branch",
        "source_artifacts",
        "target_documents",
        "evidence_claim",
        "gate_verdict",
    ]

    for entry in entries:
        for field in required_fields:
            assert field in entry, f"Missing field {field} in entry {entry.get('entry_id')}"
        assert entry["operator_branch"] == "v2-targeted-continual-alignment"
