import logging
import time
import typing
from contextlib import AbstractContextManager
from functools import lru_cache
from typing import Dict, List

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)


class log_tracing_statistics(AbstractContextManager):
    def __init__(self):
        self._root_span_cm = None
        self._span_processor = BatchSpanProcessor(LogsProfilerSpanExporter())

    def __enter__(self):
        # Retrieve/create tracer provider
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)

        # FIXME accumulation of processors
        provider.add_span_processor(self._span_processor)

        self._root_span_cm = tracer.start_as_current_span("demo-root")
        self._root_span_cm.__enter__()

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._root_span_cm is not None:
            self._root_span_cm.__exit__(exc_type, exc, tb)

        if self._span_processor is not None:
            self._span_processor.shutdown()
            self._span_processor.force_flush()

        return False


def _build_profiler_tree(spans: typing.Sequence[ReadableSpan]) -> Dict[str, List[ReadableSpan]]:
    tree = {}

    @lru_cache(maxsize=None)
    def span_id_to_name(span_id: str) -> str:
        span = next((s for s in spans if s.span_id == span_id), None)
        if span.parent is None:
            return span.name
        return span_id_to_name(span.parent.span_id) + "." + span.name

    for span in spans:
        tree.setdefault(span_id_to_name(span.span_id), []).append(span)

    return tree


class LogsProfilerSpanExporter(SpanExporter):
    def __init__(
        self,
        min_display_dur_ms=0,
        log_gaps_longer_than_ms=None,
    ):
        self.min_display_dur_s = min_display_dur_ms / 1000
        self.min_gap_dur_s = float("inf") if log_gaps_longer_than_ms is None else log_gaps_longer_than_ms / 1000

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        root_entry = _build_profiler_tree(spans)

        return SpanExportResult.SUCCESS


tracer = trace.get_tracer("torchstream.demo")

with log_tracing_statistics():
    with tracer.start_as_current_span("demo-request") as request_span:
        request_span.set_attribute("demo.request_id", "example-1")
        request_span.add_event("request.received")

        with tracer.start_as_current_span("load-model") as load_span:
            load_span.add_event("loading.start")
            time.sleep(0.05)
            load_span.add_event("loading.completed", {"result": "cached"})

        with tracer.start_as_current_span("run-model") as run_span:
            run_span.set_attribute("batch.size", 3)
            time.sleep(0.08)
            run_span.add_event("model.output", {"tokens": 128})

        request_span.add_event("request.completed")
