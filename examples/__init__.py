"""Utilities and demos that rely on optional dependencies."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterable
from typing import Optional

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter

from torchstream.analysis import telemetry

logger = logging.getLogger(__name__)

__all__ = [
    "build_sdk_tracer_provider",
    "configure_sdk_tracing",
    "demo_console_trace",
]


def build_sdk_tracer_provider(
    service_name: str = telemetry.DEFAULT_SERVICE_NAME,
    exporters: Optional[Iterable[SpanExporter]] = None,
) -> TracerProvider:
    """Create a tracer provider configured with the supplied exporters."""

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    active_exporters = list(exporters) if exporters is not None else [ConsoleSpanExporter(stream=sys.stdout)]
    for exporter in active_exporters:
        provider.add_span_processor(SimpleSpanProcessor(exporter))

    return provider


def configure_sdk_tracing(
    service_name: str = telemetry.DEFAULT_SERVICE_NAME,
    exporters: Optional[Iterable[SpanExporter]] = None,
    *,
    force: bool = False,
) -> TracerProvider:
    """Build an SDK tracer provider and register it with the telemetry helpers."""

    provider = build_sdk_tracer_provider(service_name=service_name, exporters=exporters)
    telemetry.configure_tracing(provider, force=force)
    return provider


def demo_console_trace() -> None:
    """Simple demo that records a nested span structure and prints to stdout."""

    provider = configure_sdk_tracing(service_name="torchstream-demo", force=True)
    tracer = telemetry.get_tracer("torchstream.demo", version="0.0")

    with tracer.start_as_current_span("demo-request") as request_span:
        request_span.set_attribute("demo.request_id", "example-1")
        request_span.add_event("request.received")

        with tracer.start_as_current_span("load-model") as load_span:
            load_span.add_event("loading.start")
            _simulate_work(0.05)
            load_span.add_event("loading.completed", {"result": "cached"})

        with tracer.start_as_current_span("run-model") as run_span:
            run_span.set_attribute("batch.size", 3)
            _simulate_work(0.08)
            run_span.add_event("model.output", {"tokens": 128})

        request_span.add_event("request.completed")

    force_flush = getattr(provider, "force_flush", None)
    if callable(force_flush):
        force_flush()

    telemetry.shutdown_tracing()


def _simulate_work(duration: float) -> None:
    """Busy-wait helper so spans have a noticeable duration."""

    time.sleep(duration)


if __name__ == "__main__":
    demo_console_trace()
