import logging

import torchaudio

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# FIXME! the default pad_mode="reflect" fails as it requires a minimum input size higher than modeled
n_fft = 100
transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, center=True, pad_mode="constant")
in_spec = SeqSpec(-1)
out_spec = SeqSpec(n_fft // 2 + 1, -1)

params = find_sliding_window_params(transform, in_spec, out_spec)
print(params)

# test_stream_equivalent(
#     transform,
#     SlidingWindowStream(transform, params, in_spec, out_spec),
# )
