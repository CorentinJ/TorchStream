import logging

import torch

from torchstream.patching.call_intercept import intercept_calls
from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

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

# # Normal, non-streaming inference
# import soundfile as sf
# *_, audio = next(pipeline(text, voice="af_heart"))
# sf.write("demo_audio.wav", audio, 24000)
# quit()


def trsfm(t: torch.Tensor):
    asr = t.expand((-1, 512, -1))
    F0_curve = t.repeat_interleave(2, dim=-1)[0]
    N = F0_curve
    s = t.new_zeros(1, 128)

    return pipeline.model.decoder.forward(asr, F0_curve, N, s)


def cumsum_patch_with_nan_passthrough(*args, original_fn, **kwargs):
    x = args[0]
    nan_mask = torch.isnan(x)

    x[nan_mask] = 0.0
    x = original_fn(*args, **kwargs)

    x[nan_mask] = float("nan")

    return x


def instancenorm_patch_noop(*args, original_fn, **kwargs):
    # return original_fn(*args, **kwargs)
    return args[0]


with intercept_calls("torch.nn.functional.instance_norm", instancenorm_patch_noop, pass_original_fn=True):
    with intercept_calls("torch.cumsum", cumsum_patch_with_nan_passthrough, pass_original_fn=True):
        find_sliding_window_params(
            trsfm,
            SeqSpec(1, 1, -1, device=device),  # Decoder in
            SeqSpec(1, 1, -1, device=device),  # Audio
        )
