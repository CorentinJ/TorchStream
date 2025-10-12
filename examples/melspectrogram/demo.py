import logging

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params,
)
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# n_fft = 100
# transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, center=True)
# in_spec = SeqSpec(-1)
# out_spec = SeqSpec(n_fft // 2 + 1, -1)


# Solver does not find sol with this trivial min input size transform, but does not reject the real sol
# either.
#   -> It yields sols of increasingly large context (with ki & lp growing) -> need to bound by context!
#   -> Sols get kernel rejected but the sampler is not guided any better, that's why we need context modeling


def transform(x):
    if x.shape[-1] < 30:
        raise RuntimeError()
    return x


in_spec = out_spec = SeqSpec(1, 1, -1)

params = find_sliding_window_params(
    transform, in_spec, out_spec, debug_ref_params=SlidingWindowParams(min_input_size=31)
)[0]
print(params)

test_stream_equivalent(
    transform,
    SlidingWindowStream(transform, params, in_spec, out_spec),
)
