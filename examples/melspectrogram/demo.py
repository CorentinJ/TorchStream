import logging

import torchaudio

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream
from torchstream.stream_equivalence import test_stream_equivalent

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


n_fft = 100
transform = torchaudio.transforms.Spectrogram(
    n_fft=n_fft,
    # Knobs:
    #   Set center=False to have the padding be zero
    center=True,
    #   Additionally set pad_mode="constant" to lower the minimum input size virtually inflated by reflect padding
    pad_mode="reflect",
    #   Modify hop_length to change the stride and have overlapping windows
    hop_length=30,
)
in_spec = SeqSpec(-1)
out_spec = SeqSpec(n_fft // 2 + 1, -1)


params = find_sliding_window_params(transform, in_spec, out_spec)[0]
print(params)

test_stream_equivalent(
    transform,
    SlidingWindowStream(transform, params, in_spec, out_spec),
    in_seq=in_spec.new_randn(500),
)
