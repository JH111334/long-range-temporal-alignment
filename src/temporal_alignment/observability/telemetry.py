# -*- coding: utf-8 -*-
"""
OpenTelemetry instrumentation and structured diagnostics for M0-FC40 forensic recovery.
Compliant with Arize Phoenix data model and OpenTelemetry Trace semantic conventions.
Offline-first: writes durable, schema-validated JSONL spans locally and on remote Kaggle.
"""

from __future__ import annotations

import gzip
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode


class JsonlSpanExporter(SpanExporter):
    """Deterministic, offline-first OTel SpanExporter writing compressed or plain JSONL."""

    def __init__(self, output_path: Path | str, compress: bool = False) -> None:
        self.output_path = Path(output_path)
        self.compress = compress
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.spans_written = 0

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            records = []
            for span in spans:
                ctx = span.get_span_context()
                parent_id = f"{span.parent.span_id:016x}" if span.parent else None
                trace_id = f"{ctx.trace_id:032x}"
                span_id = f"{ctx.span_id:016x}"

                # Sanitize attributes to guarantee scalar types and no secrets
                sanitized_attrs = {}
                for k, v in (span.attributes or {}).items():
                    if any(sec in k.lower() for sec in ["key", "token", "secret", "auth", "cred"]):
                        continue
                    if isinstance(v, (int, float, str, bool)):
                        sanitized_attrs[k] = v
                    elif isinstance(v, (list, tuple)):
                        sanitized_attrs[k] = [str(x) for x in v[:20]]
                    else:
                        sanitized_attrs[k] = str(v)[:200]

                events_list = []
                for ev in span.events:
                    events_list.append({
                        "name": ev.name,
                        "timestamp_ns": ev.timestamp,
                        "attributes": {ek: str(ev_val)[:200] for ek, ev_val in (ev.attributes or {}).items()},
                    })

                record = {
                    "name": span.name,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "parent_span_id": parent_id,
                    "start_time_ns": span.start_time,
                    "end_time_ns": span.end_time,
                    "status_code": span.status.status_code.name,
                    "status_description": span.status.description,
                    "attributes": sanitized_attrs,
                    "events": events_list,
                }
                records.append(record)

            line_str = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
            if self.compress or str(self.output_path).endswith(".gz"):
                with gzip.open(self.output_path, "at", encoding="utf-8") as f:
                    f.write(line_str)
            else:
                with open(self.output_path, "a", encoding="utf-8") as f:
                    f.write(line_str)

            self.spans_written += len(records)
            return SpanExportResult.SUCCESS
        except Exception:
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def setup_forensic_tracer(
    service_name: str = "temporal-m0-fc40",
    run_id: str = "forensic-001",
    output_span_file: Optional[Path | str] = None,
) -> Tuple[trace.Tracer, TracerProvider, Optional[JsonlSpanExporter]]:
    """Initialize OTel Tracer with Resource metadata and local JSONL exporter."""
    resource = Resource.create({
        "service.name": service_name,
        "service.version": "7ae557604adf",
        "telemetry.sdk.language": "python",
        "deployment.environment": "kaggle" if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") else "local",
        "forensic.run_id": run_id,
    })

    provider = TracerProvider(resource=resource)
    exporter = None
    if output_span_file:
        exporter = JsonlSpanExporter(output_span_file)
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)

    tracer = provider.get_tracer(service_name)
    return tracer, provider, exporter


def verify_synthetic_roundtrip(output_dir: Path) -> Dict[str, Any]:
    """Execute a 3-span synthetic round trip (parent -> step -> eval) and verify schema."""
    out_file = output_dir / "synthetic_trace_spans.jsonl"
    tracer, provider, exporter = setup_forensic_tracer(
        service_name="temporal-m0-fc40-synthetic",
        run_id="synth-001",
        output_span_file=out_file,
    )

    with tracer.start_as_current_span("M0_FC40.recovery") as parent_span:
        parent_span.set_attribute("experiment.id", "SYNTH_ROUNDTRIP")
        parent_span.set_attribute("target.contract", "GENERATIVE_STRICT_PARSE_V1")

        with tracer.start_as_current_span("canary.train.step") as step_span:
            step_span.set_attribute("step.index", 0)
            step_span.set_attribute("loss.masked_ce", 3.1415)
            step_span.set_attribute("step.is_finite", True)

        with tracer.start_as_current_span("eval.strict_generation") as eval_span:
            eval_span.set_attribute("eval.cases_evaluated", 1)
            eval_span.set_attribute("eval.strict_correct", True)

    provider.shutdown()

    # Read back and verify spans
    lines = out_file.read_text(encoding="utf-8").splitlines()
    spans_data = [json.loads(line) for line in lines if line.strip()]

    # Verification criteria
    has_parent = any(s["name"] == "M0_FC40.recovery" for s in spans_data)
    has_step = any(s["name"] == "canary.train.step" for s in spans_data)
    has_eval = any(s["name"] == "eval.strict_generation" for s in spans_data)

    parent_span_id = next(s["span_id"] for s in spans_data if s["name"] == "M0_FC40.recovery")
    step_parent = next(s["parent_span_id"] for s in spans_data if s["name"] == "canary.train.step")
    eval_parent = next(s["parent_span_id"] for s in spans_data if s["name"] == "eval.strict_generation")

    parentage_ok = (step_parent == parent_span_id) and (eval_parent == parent_span_id)

    status = "OTEL_LOCAL_ONLY_VERIFIED" if (has_parent and has_step and has_eval and parentage_ok) else "EXPORT_FAILED"

    return {
        "telemetry_sink_status": status,
        "spans_count": len(spans_data),
        "parentage_verified": parentage_ok,
        "trace_id_sample": spans_data[0]["trace_id"] if spans_data else None,
        "output_file": str(out_file),
    }
