import logging

from opentelemetry import trace

from examples.resources.bigvgan.bigvgan import BigVGAN
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

logging.basicConfig(level=logging.INFO)

device = "cuda"


def load_bigvgan() -> BigVGAN:
    model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model


model = load_bigvgan()

in_spec = SeqSpec(1, model.h.num_mels, -1, device=device)
out_spec = SeqSpec(1, 1, -1, device=device)
params = find_sliding_window_params(
    model,
    in_spec,
    out_spec,
    max_in_out_seq_size=1_000_000,
)
print(params)

# params = SlidingWindowParams(
#     kernel_size_in=35,
#     stride_in=1,
#     left_pad=17,
#     right_pad=17,
#     kernel_size_out=418,
#     stride_out=256,
#     left_out_trim=81,
#     right_out_trim=81,
# )

# test_stream_equivalent(
#     bigvgan,
#     SlidingWindowStream(bigvgan, params, in_spec, out_spec),
#     in_step_sizes=(7, 4, 12) + (1,) * 100 + (17, 9),
#     throughput_check_max_delay=params.out_trim,
# )
# quit()
