import soundfile as sf
import torch
from kokoro import KPipeline

from torchstream.call_intercept import intercept

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

# uv pip install pip
# .venv\Scripts\python.exe -m spacy download en_core_web_sm
# spacy.load("en_core_web_sm")


pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""

# Normal, non-streaming inference
# *_, audio = next(pipeline(text, voice="af_heart"))
# sf.write("demo_audio.wav", audio, 24000)


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


with intercept("kokoro.istftnet.Decoder.forward") as calls:
    *_, audio = next(pipeline(text, voice="af_heart"))
    sf.write("demo_audio.wav", audio, 24000)
