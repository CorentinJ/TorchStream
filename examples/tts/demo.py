import logging

import torch

from torchstream.sequence.seq_spec import SeqSpec
from torchstream.sliding_window.sliding_window_params_solver import find_sliding_window_params

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
# import soundfile as sf
# *_, audio = next(pipeline(text, voice="af_heart"))
# sf.write("demo_audio.wav", audio, 24000)
# quit()


# Step 1: because kokoro is a third party library, we'll avoid modifying it's source code directly. Instead,
# we'll monkey-patch the method where streaming can happen and work there.
def mod_decoder_forward(self, asr, F0_curve, N, s):
    F0 = self.F0_conv(F0_curve.unsqueeze(1))
    N = self.N_conv(N.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = self.encode(x, s)
    asr_res = self.asr_res(asr)
    res = True
    for block in self.decode:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    x = self.generator(x, s, F0_curve)
    return x


def trsfm(x: torch.Tensor):
    asr = x.expand((-1, 512, -1))
    F0_curve = x.repeat_interleave(2, dim=-1)[0]
    N = F0_curve
    s = x.new_zeros(1, 128)

    # return pipeline.model.decoder.forward(asr, F0_curve, N, s)

    self = pipeline.model.decoder

    F0 = self.F0_conv(F0_curve.unsqueeze(1))
    N = self.N_conv(N.unsqueeze(1))
    x = torch.cat([asr, F0, N], axis=1)
    x = self.encode(x, s)
    asr_res = self.asr_res(asr)
    res = True
    for block in self.decode[:3]:
        if res:
            x = torch.cat([x, asr_res, F0, N], axis=1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    # x = self.generator(x, s, F0_curve)

    return x


find_sliding_window_params(
    trsfm,
    SeqSpec(1, 1, -1, device=device),  # Decoder in
    SeqSpec(1, 1024, -1, device=device),  # Decoder out
    # SeqSpec(1, 1, -1, device=device), # Audio
)

# *_, audio = next(pipeline(text, voice="af_heart"))
# sf.write("demo_audio.wav", audio, 24000)

# TODO
# with intercept("kokoro.istftnet.Decoder.forward") as calls:
