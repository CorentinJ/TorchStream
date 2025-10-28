import soundfile as sf
import torch
from kokoro import KPipeline

# Change the tensor repr to show the shape and device, which saves a lot of time for debugging
old_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda t: f"{tuple(t.shape)} {str(t.dtype).replace('torch.', '')} {str(t.device)} {old_repr(t)}"

# uv pip install pip
# .venv\Scripts\python.exe -m spacy download en_core_web_sm
# spacy.load("en_core_web_sm")


pipeline = KPipeline(lang_code="en-us", repo_id="hexgrad/Kokoro-82M")
text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters, wow!. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient. With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""
generator = pipeline(text, voice="af_heart")
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    # display(Audio(data=audio, rate=24000, autoplay=i == 0))
    sf.write(f"{i}.wav", audio, 24000)
