import logging

from opentelemetry import trace
from torch import nn

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO)

trsfm = nn.ConvTranspose1d(1, 1, kernel_size=2, stride=1)
find_sliding_window_params(trsfm, SeqSpec(1, 1, -1), max_equivalent_sols=10)


# (ki=1, si=1, lp=0, rp=0, ko=2, so=1, lt=0, rt=0)
# (ki=2, si=1, lp=1, rp=1, ko=1, so=1, lt=0, rt=0)
