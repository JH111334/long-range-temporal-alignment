#!/usr/bin/env python3
"""Verification script for OpenTelemetry / Phoenix Fail-Open Tracing in Stage 03."""

from __future__ import annotations

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from temporal_alignment.causal_evaluation import get_global_tracer


def verify_otel_phoenix_tracing() -> None:
    project_root = Path(__file__).resolve().parent.parent
    test_trace_dir = project_root / "artifacts" / "test_traces"
    test_trace_dir.mkdir(parents=True, exist_ok=True)

    tracer = get_global_tracer(
        service_name="stage03-verifier",
        project_name="verification-test",
    )

    print(f"Tracer initialized. Collector status: {tracer.collector_status}")

    # Simulate full Stage 03 span tree
    with tracer.span("stage03.run", {"run_id": "TEST_VERIFY_RUN"}) as root_span:
        with tracer.span("stage03.preflight", {"check": "dependencies_ok"}):
            time_ns = 100

        with tracer.span("stage03.condition", {"condition": "TEST_ARM"}):
            with tracer.span("stage03.counterfactual_pair", {"pair_id": "TEST_PAIR_01"}):
                with tracer.span("temporal.parse", {"year": 1980}):
                    pass
                with tracer.span("temporal.encode", {"dim": 40}):
                    pass
                with tracer.span("temporal.inject", {"position": 14}):
                    pass
                with tracer.span("temporal.core", {"rank": 72}):
                    pass
                with tracer.span("adapter.delta", {"layer": 14}):
                    pass
                with tracer.span("model.forward", {"tokens": 24}):
                    pass
                with tracer.span("model.hidden_probe", {"layer": 14, "pos": 14}):
                    pass
                with tracer.span("model.answer_score", {"logit_delta": 0.05}):
                    pass
                with tracer.span("eval.semantic_pair", {"both_correct": True}):
                    pass

        with tracer.span("stage03.aggregate", {"total_evaluated": 1}):
            pass
        with tracer.span("stage03.writeback", {"target": "TEST_LOG"}):
            pass

    # Export to local directory
    receipt = tracer.export_local_traces(test_trace_dir)
    print("Trace export completed:")
    print(json.dumps(receipt, indent=2))

    spans_file = test_trace_dir / "OTEL_SPANS.jsonl"
    with open(spans_file, "r", encoding="utf-8") as f:
        spans_count = sum(1 for line in f if line.strip())

    print(f"Successfully recorded and saved {spans_count} spans.")
    assert spans_count >= 13, f"Expected at least 13 spans in tree, got {spans_count}"


if __name__ == "__main__":
    verify_otel_phoenix_tracing()
