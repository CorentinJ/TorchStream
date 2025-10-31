"""OpenTelemetry API helpers used across TorchStream.

The TorchStream library itself only depends on the OpenTelemetry API.  Any SDK
initialisation happens in optional demo code so that importing the library does
not require ``opentelemetry-sdk``.  Callers can configure tracing by creating a
provider (for example from the SDK) and passing it to :func:`configure_tracing`.
Subsequent calls to :func:`get_tracer` will re-use the global provider.
"""

from __future__ import annotations

import logging
from typing import Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer, TracerProvider

__all__ = ["DEFAULT_SERVICE_NAME", "configure_tracing", "get_tracer", "shutdown_tracing"]

logger = logging.getLogger(__name__)

DEFAULT_SERVICE_NAME = "torchstream"
_configured_provider: Optional[TracerProvider] = None


def configure_tracing(provider: TracerProvider, *, force: bool = False) -> TracerProvider:
    """Install ``provider`` as the global tracer provider used by TorchStream.

    Parameters
    ----------
    provider:
        A fully configured tracer provider instance created by the caller.
    force:
        When ``True`` replaces a previously configured provider, shutting it down if possible.
    """

    global _configured_provider

    if provider is _configured_provider:
        return provider

    if _configured_provider is not None and not force:
        logger.debug("Tracer provider already configured; skipping reconfiguration.")
        return _configured_provider

    if _configured_provider is not None:
        _shutdown_provider(_configured_provider)

    trace.set_tracer_provider(provider)
    _configured_provider = provider
    return provider


def get_tracer(name: str, version: str | None = None) -> Tracer:
    """Return a tracer for the given instrumentation scope."""

    return trace.get_tracer(name, version)


def shutdown_tracing() -> None:
    """Best-effort shutdown of the provider configured via :func:`configure_tracing`."""

    global _configured_provider

    if _configured_provider is None:
        return

    _shutdown_provider(_configured_provider)
    _configured_provider = None


def _shutdown_provider(provider: TracerProvider) -> None:
    """Attempt to flush and shut down a tracer provider without SDK imports."""

    try:
        force_flush = getattr(provider, "force_flush", None)
        if callable(force_flush):
            force_flush()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug("Force flush failed during tracer shutdown: %s", exc)

    try:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Tracer provider shutdown raised an error: %s", exc)
