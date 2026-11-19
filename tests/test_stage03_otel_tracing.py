"""Unit tests for Stage 03 OpenTelemetry Fail-Open Tracing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from temporal_alignment.causal_evaluation.otel_tracing import (
    FailOpenOTLPSpanExporter,
    LocalMemorySpanExporter,
    Stage03Tracer,
)


def test_stage03_tracer_fail_open_lifecycle(tmp_path: Path):
    tracer = Stage03Tracer(
        service_name="test-service",
        project_name="test-project",
        endpoint="http://127.0.0.1:59999/v1/traces",  # non-existent endpoint
        fail_open=True,
    )

    # Verify fail-open status
    assert tracer.collector_status == "LOCAL_FALLBACK"

    # Execute span tree
    with tracer.span("stage03.run", {"run_id": "UNIT_TEST_RUN"}) as root_span:
        trace_id, span_id = tracer.get_current_ids()
        assert len(trace_id) == 32
        assert len(span_id) == 16
        assert trace_id != "0" * 32

        with tracer.span("stage03.condition", {"arm": "A0_BASE"}):
            with tracer.span("model.forward", {"tokens": 10}):
                pass

    # Export to tmp_path
    index = tracer.export_local_traces(tmp_path)
    assert index["total_spans_recorded"] == 3
    assert index["collector_status"] == "LOCAL_FALLBACK"

    spans_file = tmp_path / "OTEL_SPANS.jsonl"
    assert spans_file.exists()

    records = []
    with open(spans_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    assert len(records) == 3
    names = {r["name"] for r in records}
    assert names == {"stage03.run", "stage03.condition", "model.forward"}

    # Validate parent-child relation
    root_rec = next(r for r in records if r["name"] == "stage03.run")
    cond_rec = next(r for r in records if r["name"] == "stage03.condition")
    assert cond_rec["parent_span_id"] == root_rec["span_id"]
