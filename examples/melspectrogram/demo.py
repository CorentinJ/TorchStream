import logging

import torchaudio

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import (
    find_sliding_window_params,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


# FIXME! the default pad_mode="reflect" fails as it requires a minimum input size higher than modeled
transform = torchaudio.transforms.Spectrogram(n_fft=800, center=True, pad_mode="constant")
in_spec = SeqSpec((-1,))
out_spec = SeqSpec((401, -1))

# test_stream_equivalent(
#     bigvgan,
#     SlidingWindowStream(bigvgan, params, in_spec, out_spec),
#     in_step_sizes=(7, 4, 12) + (1,) * 100 + (17, 9),
#     throughput_check_max_delay=params.out_trim,
# )
# quit()

params = find_sliding_window_params(transform, in_spec, out_spec)
print(params)
