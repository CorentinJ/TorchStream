import logging
from functools import partial

import torch
from kokoro.istftnet import Decoder

from torchstream.exception_signature import DEFAULT_ZERO_SIZE_EXCEPTIONS
from torchstream.patching.call_intercept import exit_early, intercept_calls
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_stream import SlidingWindowParams, SlidingWindowStream

# Setup logging to see the solver's message
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

decoder = Decoder(512, 128, 80, (3, 7, 11), (10, 6), 512, [(1, 3, 5)] * 3, (20, 12), 20, 5)
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
audio_out_spec = SeqSpec(1, 1, -1, device=device)

decoder_trsfm = partial(decoder, s=torch.randn(1, 128))

zero_size_exception_signatures = DEFAULT_ZERO_SIZE_EXCEPTIONS + [
    (ValueError, "Expected more than 1 spatial element when training")
]

inps = decoder_in_spec.new_randn_arrays(830)


# print(wip(*inps))
# quit()


def cumsum_patch_noop(*args, original_fn, **kwargs):
    # print("cumsum")
    # return original_fn(*args, **kwargs)
    return args[0]


def instancenorm_patch_noop(*args, original_fn, **kwargs):
    # print("norm")
    # return original_fn(*args, **kwargs)
    return args[0]


with intercept_calls("torch.nn.functional.instance_norm", instancenorm_patch_noop, pass_original_fn=True):
    # with intercept_calls("torch.cumsum", cumsum_patch_noop, pass_original_fn=True, store_in_out=True) as interceptor:
    # decoder_trsfm(*inps)
    # quit()

    # sli_params = find_sliding_window_params(
    #     exit_early,
    #     decoder_in_spec,
    #     SeqSpec(1, -1, 9, device=device),
    #     zero_size_exception_signatures=zero_size_exception_signatures,
    # )[0]
    # print(sli_params)

    # Let's see what the model gets for input when it arrives at the cumsum
    dec_trsfm_cumsum_exit = exit_early(decoder_trsfm, target_to_exit_on="torch.cumsum", out_proc_fn=lambda x, dim: x)
    ref_cumsum_in = dec_trsfm_cumsum_exit(*inps)
    print(ref_cumsum_in)

    # And what it gets when streaming with the same sliding window parameters we found earlier
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
    stream = SlidingWindowStream(decoder_trsfm, sli_params, decoder_in_spec, audio_out_spec)
    with intercept_calls("torch.cumsum", store_in_out=True) as interceptor:
        stream.forward_chunks(*inps, chunk_size=120)
        stream_cumsum_in_outs = interceptor.calls_in_out

    comp = SeqSpec(1, -1, 9, device=device).new_empty_sequence()
    for i, (in_args, in_kwargs, out) in enumerate(stream_cumsum_in_outs):
        print(in_args[0].shape)

        if i > 0:
            comp.feed(in_args[0][:, 56:])
        else:
            comp.feed(in_args[0])

    3 + 2
