import logging
from functools import partial

import torch

from torchstream.patching.call_intercept import intercept_calls
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_stream import SlidingWindowParams, SlidingWindowStream

# Setup logging to see the solver's message
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

# decoder = Decoder(512, 128, 80, (3, 7, 11), (10, 6), 512, [(1, 3, 5)] * 3, (20, 12), 20, 5)

from kokoro import KPipeline

pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
decoder = pipeline.model.decoder

sli_params = SlidingWindowParams(
    kernel_size_in=28,
    stride_in=1,
    left_pad=14,
    right_pad=14,
    kernel_size_out=675,
    stride_out=600,
    left_out_trim=185,
    right_out_trim=490,
)
device = torch.device("cpu")

decoder_in_spec = SeqSpec(
    # asr
    (1, 512, -1, device),
    # f0_curve (twice the time resolution of asr -> we put -2 to scale accordingly)
    (1, -2, device),
    # n (same as above)
    (1, -2, device),
    # s is a fixed input, it does not fit as sequential data
)
# Audio is 1-dimensional, but is output with the batch & channel dimensions
decoder_out_spec = SeqSpec(1, 1, -1, device=device)

decoder_trsfm = partial(decoder, s=torch.randn(1, 128))

decoder_input = decoder_in_spec.new_randn_sequence(830)


def get_streaming_cumsum():
    accum_value = 0.0

    def streaming_cumsum(x, dim, original_fn):
        nonlocal accum_value
        out = original_fn(x, dim=dim) + accum_value

        assert dim == 1
        accum_value = out[:, -2 * sli_params.streaming_context_size - 1, :]

        return out

    return streaming_cumsum


def get_streaming_instance_norm():
    unique_locations = set()

    def streaming_instance_norm(instnorm_obj, x):
        unique_locations.add(instnorm_obj)
        print(f"Total of {len(unique_locations)} unique instance_norm call locations detected.")
        return x

    return streaming_instance_norm


with intercept_calls("torch.cumsum", get_streaming_cumsum(), pass_original_fn=True):
    with intercept_calls(
        "torch.nn.modules.instancenorm.InstanceNorm1d.forward",
        get_streaming_instance_norm(),
    ):
        stream = SlidingWindowStream(decoder_trsfm, sli_params, decoder_in_spec, decoder_out_spec)
        stream(decoder_input[100:200])
        print("--- Second call ---")
        stream(decoder_input[100:200])
        # decoder_input.stream_apply(decoder_trsfm, sli_params, chunk_size=40, out_spec=decoder_out_spec)
