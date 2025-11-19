import logging

from opentelemetry import trace

from torchstream.sliding_window.sliding_window_params import SlidingWindowParams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO)


s = SlidingWindowParams(5, 3, right_pad=2)
print(s.output_delays)
