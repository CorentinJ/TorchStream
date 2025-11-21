import logging

from opentelemetry import trace

from torchstream import SlidingWindowParams

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO)


s = SlidingWindowParams(kernel_size_in=5, stride_in=3, kernel_size_out=3, stride_out=2)
for c in s.get_inverse_kernel_map(in_len=20):
    print(c)
