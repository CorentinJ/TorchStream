import logging
from functools import partial

import torch

from torchstream.patching.call_intercept import intercept_calls
from torchstream.sequence.sequence import SeqSpec
from torchstream.sliding_window.sliding_window_params import SlidingWindowParams
from torchstream.sliding_window.sliding_window_stream import SlidingWindowStream

# Setup logging to see the solver's message
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

# uv pip install pip
# .venv\Scripts\python.exe -m spacy download en_core_web_sm
# spacy.load("en_core_web_sm")

from kokoro import KPipeline

pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
device = pipeline.model.device
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""

# Normal, non-streaming inference
import soundfile as sf

with intercept_calls("kokoro.istftnet.Decoder.forward", store_in_out=True) as interceptor:
    *_, audio = next(pipeline(text, voice="af_heart"))
    (decoder, ref_asr, ref_f0_curve, ref_n, ref_s), _, ref_audio = interceptor.call_in_outs[0]
sf.write("demo_audio.wav", audio, 24000)


decoder_in_spec = SeqSpec(
    # asr
    (1, 512, -1, device),
    # f0_curve (twice the time resolution of asr -> we put -2 to scale accordingly)
    (1, -2, device),
    # n (same as above)
    (1, -2, device),
    # s is a fixed input of size (1, 128), it does not fit as sequential data
)
# Audio is 1-dimensional, but is output with the batch & channel dimensions
audio_out_spec = SeqSpec(1, 1, -1, device=device)

decoder_trsfm = partial(pipeline.model.decoder.forward, s=ref_s)


def cumsum_patch_with_nan_passthrough(x, dim, original_fn):
    nan_mask = torch.isnan(x)

    x[nan_mask] = 0.0
    x = original_fn(x, dim=dim)

    x[nan_mask] = float("nan")

    return x


def instancenorm_patch_noop(*args, original_fn, **kwargs):
    return original_fn(*args, **kwargs)
    # return args[0]


with intercept_calls("torch.nn.functional.instance_norm", instancenorm_patch_noop, pass_original_fn=True):
    with intercept_calls(
        "torch.cumsum", cumsum_patch_with_nan_passthrough, pass_original_fn=True, store_in_out=True
    ) as cumsum_interceptor:
        *_, audio = next(pipeline(text, voice="af_heart"))

        # sf.write("demo_audio_patched.wav", audio, 24000)
        # quit()

        # sli_params = find_sliding_window_params(decoder_trsfm, decoder_in_spec, audio_out_spec)[0]

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
        audio = stream.forward_all_chunks(ref_asr, ref_f0_curve, ref_n, chunk_size=100).data[0]
        sf.write("demo_audio_streamed.wav", audio[0, 0], 24000)
