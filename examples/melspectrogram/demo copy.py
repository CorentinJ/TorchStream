import logging

import torchaudio

from torchstream import (
    SeqSpec,
    SlidingWindowParams,
    SlidingWindowStream,
    test_stream_equivalent,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


n_fft = 512
hop_length = 64
n_mels = 80
transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=48000,
    n_fft=n_fft,
    center=False,
    hop_length=hop_length,
    n_mels=n_mels,
    f_min=50.0,
    f_max=0.5 * 48000,
)


in_spec = SeqSpec(-1)
out_spec = SeqSpec(n_mels, -1)
# solutions = find_sliding_window_params(
#     transform,
#     in_spec,
#     out_spec,
#     zero_size_exception_signatures=[(RuntimeError, "expected 0 < n_fft <")],
# )
# print(solutions)
# params = solutions[0]
params = SlidingWindowParams(
    kernel_size_in=64,
    stride_in=64,
    left_pad=0,
    right_pad=0,
    kernel_size_out=8,
    stride_out=1,
    left_out_trim=7,
    right_out_trim=7,
)

test_stream_equivalent(
    transform,
    SlidingWindowStream(transform, params, in_spec, out_spec),
    in_data=in_spec.new_randn_arrays(params.min_input_size * 2),
    throughput_check_max_delay=params.right_out_trim,
    atol=1e-3,
)
