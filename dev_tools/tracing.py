import logging
import typing
from contextlib import AbstractContextManager
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
tracer = trace.get_tracer(__name__)


class log_tracing_profile(AbstractContextManager):
    def __init__(self, name: str = "ctx-manager"):
        self.name = name
        self._root_span_cm = None
        self._span_processor = BatchSpanProcessor(
            LogsProfilerSpanExporter(name),
            # Quick hack to ensure we export all spans in one go
            max_queue_size=int(1e10),
            schedule_delay_millis=1e8,
            max_export_batch_size=int(1e10),
        )

    def __enter__(self):
        # Retrieve/create tracer provider
        provider = trace.get_tracer_provider()
        if not hasattr(provider, "add_span_processor"):
            provider = TracerProvider()
            trace.set_tracer_provider(provider)

        # FIXME accumulation of processors
        provider.add_span_processor(self._span_processor)

        self._root_span_cm = tracer.start_as_current_span(self.name)
        self._root_span_cm.__enter__()

        return self

    def __exit__(self, exc_type, exc, tb):
        if self._root_span_cm is not None:
            self._root_span_cm.__exit__(exc_type, exc, tb)

        if self._span_processor is not None:
            self._span_processor.shutdown()
            self._span_processor.force_flush()

        return False


def _group_spans_by_hierarchy(spans: typing.Sequence[ReadableSpan]) -> Dict[str, List[ReadableSpan]]:
    tree = {}

    @lru_cache(maxsize=None)
    def span_id_to_name(span_id: str) -> str:
        span = next((s for s in spans if s.context.span_id == span_id), None)
        span_name = span.name.replace("/", "_")
        if span.parent is None:
            return span_name
        # TODO: use tuples instead of str concat, we're not displaying these anyway
        return span_id_to_name(span.parent.span_id) + "/" + span_name

    for span in spans:
        tree.setdefault(span_id_to_name(span.context.span_id), []).append(span)

    return tree


class LogsProfilerSpanExporter(SpanExporter):
    def __init__(
        self,
        root_span_name: str,
        min_display_dur_ms=0,
        log_gaps_longer_than_ms=None,
    ):
        self.root_span_name = root_span_name
        self.min_display_dur_s = min_display_dur_ms / 1000
        self.min_gap_dur_s = float("inf") if log_gaps_longer_than_ms is None else log_gaps_longer_than_ms / 1000

    def export(self, spans: typing.Sequence[ReadableSpan]) -> SpanExportResult:
        name2spans = _group_spans_by_hierarchy(spans)

        def build_log_table(span_name) -> List[Tuple[str, ...]]:
            depth = span_name.count("/") + 1
            line = []
            line.append(" " + "   " * (depth - 1) + span_name.split("/")[-1])
            line.append(f"  ({len(name2spans[span_name])}x)")
            durations_ms = [(s.end_time - s.start_time) / 1e6 for s in name2spans[span_name]]
            line.extend(("  ", "total: ", f"{np.sum(durations_ms):.0f}ms"))
            if len(durations_ms) > 1:
                line.extend(("  mean: ", f"{np.mean(durations_ms):.0f}ms"))
                line.extend(("  std: ", f"{np.std(durations_ms):.0f}ms"))
            else:
                line.extend([""] * 4)

            children_names = [
                name for name in name2spans.keys() if name.startswith(span_name + "/") and name.count("/") == depth
            ]

            return [tuple(line)] + [line for child_name in children_names for line in build_log_table(child_name)]

        table = build_log_table(self.root_span_name)

        # Align all columns
        max_lens = [max(len(row[i]) for row in table) for i in range(len(table[0]))]
        log_message = "\n".join(
            "".join(((row[i].rjust if i > 0 else row[i].ljust)(max_lens[i]) for i in range(len(row)))) for row in table
        )

        logger.info("Profile from tracing\n" + log_message)

        return SpanExportResult.SUCCESS
