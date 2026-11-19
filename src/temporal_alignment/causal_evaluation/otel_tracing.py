"""OpenTelemetry and Arize Phoenix Tracing Module with Fail-Open Resilience.

Provides hierarchical span tracing across the Stage 03 temporal causality pipeline:
stage03.run -> stage03.preflight -> stage03.condition -> stage03.counterfactual_pair
-> [temporal.*, adapter.*, model.*, eval.*] -> stage03.aggregate -> stage03.writeback.

Ensures strict fail-open behavior: failures in remote OTLP export do not halt local evaluation.
Local span records with W3C trace_id (32-hex) and span_id (16-hex) are persisted deterministically.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger("temporal_alignment.otel_tracing")


def _sanitize_attribute_value(val: Any) -> Any:
    """Sanitize attribute values for OpenTelemetry compatibility."""
    if isinstance(val, (int, float, str, bool)):
        return val
    if isinstance(val, (list, tuple)):
        sanitized_list = []
        for x in val:
            if isinstance(x, (int, float, str, bool)):
                sanitized_list.append(x)
            else:
                sanitized_list.append(str(x))
        return sanitized_list
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


@dataclass
class LocalSpanRecord:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    start_time_ns: int
    end_time_ns: int
    duration_ms: float
    status: str
    attributes: Dict[str, Any] = field(default_factory=dict)


class LocalMemorySpanExporter(SpanExporter):
    """Retains spans in-memory for deterministic local JSONL persistence."""

    def __init__(self) -> None:
        self.spans: List[ReadableSpan] = []

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def clear(self) -> None:
        self.spans.clear()

    def get_records(self) -> List[LocalSpanRecord]:
        records: List[LocalSpanRecord] = []
        for sp in self.spans:
            ctx = sp.get_span_context()
            parent = sp.parent
            duration_ms = (sp.end_time - sp.start_time) / 1e6 if sp.end_time and sp.start_time else 0.0
            rec = LocalSpanRecord(
                trace_id=format(ctx.trace_id, "032x"),
                span_id=format(ctx.span_id, "016x"),
                parent_span_id=format(parent.span_id, "016x") if parent else None,
                name=sp.name,
                start_time_ns=sp.start_time or 0,
                end_time_ns=sp.end_time or 0,
                duration_ms=round(duration_ms, 4),
                status=sp.status.status_code.name,
                attributes=dict(sp.attributes) if sp.attributes else {},
            )
            records.append(rec)
        return records


class FailOpenOTLPSpanExporter(SpanExporter):
    """Wraps standard OTLP HTTP exporter with safe fail-open error trapping."""

    def __init__(self, endpoint: str, is_enabled: bool = True, timeout_seconds: float = 2.0) -> None:
        self.endpoint = endpoint
        self.is_enabled = is_enabled
        self.timeout = timeout_seconds
        self.delegate: Optional[Any] = None
        self.export_failed_count = 0
        self.export_success_count = 0

        if self.is_enabled:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                self.delegate = OTLPSpanExporter(endpoint=endpoint, timeout=timeout_seconds)
            except Exception as e:
                logger.warning("Could not initialize OTLPSpanExporter: %s (fail-open fallback active)", e)

    def export(self, spans: List[ReadableSpan]) -> SpanExportResult:
        if not self.is_enabled or not self.delegate:
            self.export_failed_count += len(spans)
            return SpanExportResult.SUCCESS

        try:
            res = self.delegate.export(spans)
            if res == SpanExportResult.SUCCESS:
                self.export_success_count += len(spans)
            else:
                self.export_failed_count += len(spans)
            return res
        except Exception as e:
            self.export_failed_count += len(spans)
            logger.debug("Fail-open caught OTLP export exception: %s", e)
            return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        if self.delegate:
            try:
                self.delegate.shutdown()
            except Exception:
                pass


class Stage03Tracer:
    """Central tracing session managing OpenTelemetry and local span ledger."""

    def __init__(
        self,
        service_name: str = "temporal-alignment-stage03",
        project_name: str = "stage03-causal-bottleneck-sweep",
        endpoint: str = "http://localhost:6006/v1/traces",
        fail_open: bool = True,
    ) -> None:
        self.service_name = service_name
        self.project_name = project_name
        self.endpoint = endpoint
        self.fail_open = fail_open
        self.collector_status = self._probe_collector()

        resource = Resource.create(
            {
                "service.name": service_name,
                "project.name": project_name,
                "telemetry.sdk.language": "python",
            }
        )

        self.provider = TracerProvider(resource=resource)
        self.memory_exporter = LocalMemorySpanExporter()
        self.provider.add_span_processor(SimpleSpanProcessor(self.memory_exporter))

        is_connected = (self.collector_status == "CONNECTED")
        self.otlp_exporter = FailOpenOTLPSpanExporter(endpoint=endpoint, is_enabled=is_connected)
        self.provider.add_span_processor(SimpleSpanProcessor(self.otlp_exporter))

        self.tracer = self.provider.get_tracer(service_name)

    def _probe_collector(self) -> str:
        """Probe whether Arize Phoenix collector is actively listening."""
        try:
            req = urllib.request.Request(self.endpoint, method="HEAD")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status in (200, 404, 405):
                    return "CONNECTED"
        except Exception:
            pass
        return "LOCAL_FALLBACK"

    def get_current_ids(self) -> Tuple[str, str]:
        """Return (trace_id, span_id) for current execution context."""
        current_span = trace.get_current_span()
        ctx = current_span.get_span_context()
        if ctx.is_valid:
            return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
        return "0" * 32, "0" * 16

    @contextlib.contextmanager
    def span(
        self,
        name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[trace.Span, None, None]:
        """Context manager to open a hierarchical tracing span."""
        sanitized_attrs = {}
        if attributes:
            for k, v in attributes.items():
                sanitized_attrs[k] = _sanitize_attribute_value(v)

        with self.tracer.start_as_current_span(name, attributes=sanitized_attrs) as sp:
            try:
                yield sp
                sp.set_status(Status(StatusCode.OK))
            except Exception as exc:
                sp.set_status(Status(StatusCode.ERROR, str(exc)))
                sp.record_exception(exc)
                raise

    def export_local_traces(self, output_dir: Path) -> Dict[str, Any]:
        """Write recorded spans and index receipt to local directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        spans_file = output_dir / "OTEL_SPANS.jsonl"
        index_file = output_dir / "TRACE_INDEX.json"

        records = self.memory_exporter.get_records()
        with open(spans_file, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")

        trace_index = {
            "service_name": self.service_name,
            "project_name": self.project_name,
            "collector_endpoint": self.endpoint,
            "collector_status": self.collector_status,
            "total_spans_recorded": len(records),
            "otlp_export_success_count": self.otlp_exporter.export_success_count,
            "otlp_export_failed_count": self.otlp_exporter.export_failed_count,
            "trace_files": {
                "spans_jsonl": spans_file.name,
                "spans_count": len(records),
            },
        }

        with open(index_file, "w", encoding="utf-8") as f:
            json.dump(trace_index, f, indent=2, ensure_ascii=False)

        return trace_index

    def shutdown(self) -> None:
        """Shutdown tracer provider safely."""
        self.provider.shutdown()


_GLOBAL_TRACER: Optional[Stage03Tracer] = None


def get_global_tracer(
    service_name: str = "temporal-alignment-stage03",
    project_name: str = "stage03-causal-bottleneck-sweep",
    endpoint: str = "http://localhost:6006/v1/traces",
) -> Stage03Tracer:
    """Singleton getter for Stage 03 tracer."""
    global _GLOBAL_TRACER
    if _GLOBAL_TRACER is None:
        _GLOBAL_TRACER = Stage03Tracer(
            service_name=service_name,
            project_name=project_name,
            endpoint=endpoint,
        )
    return _GLOBAL_TRACER
