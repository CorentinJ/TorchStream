# TorchStream

TorchStream is a library to help ML developers stream PyTorch models **without retraining nor rewriting them**, 
in order to reduce their latency or use them in live applications.

TorchStream comes with [a website of live examples](https://excellent-diandra-corentinj-63561158.koyeb.app/).


## Installation
Install as a package (any OS, CUDA optional):
```bash
(uv) pip install torchstream-lib
```

Install as a project, to run the streamlit examples yourself (some examples require CUDA):
```bash
git clone https://github.com/CorentinJ/TorchStream
cd TorchStream
uv run --group demos streamlit run examples
```

If you don't have `uv` yet:
```bash
# On Windows:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
# On Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Alternatively, on any platform if you have pip installed you can do
pip install -U uv
```


## Overview
TorchStream offers a set of tools to help you stream complex neural networks and other sequence to sequence transforms. 

The example below requires cloning the project and installing the demos dependencies (`uv sync --group demos`). It streams BigVGAN, a state of the art neural vocoder:

```python
import logging

import librosa
import torch

from examples.resources.bigvgan.bigvgan import BigVGAN
from examples.resources.bigvgan.meldataset import get_mel_spectrogram
from torchstream import SeqSpec, SlidingWindowStream, find_sliding_window_params

logging.basicConfig(level=logging.INFO)

device = "cuda"
model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x").eval().to(device)
model.remove_weight_norm()

# Get a sample mel spectrogram input
wave, sample_rate = librosa.load(librosa.ex("libri1"), sr=model.h.sampling_rate)
mel = get_mel_spectrogram(torch.from_numpy(wave).unsqueeze(0), model.h).to(device)

# Specify the model's input format: a melspectrogram
in_spec = SeqSpec(1, model.h.num_mels, -1, device=device)
# Output format: an audio waveform
out_spec = SeqSpec(1, 1, -1, device=device)

# Use TorchStream's solver to find the sliding window parameters of BigVGAN
sli_params = find_sliding_window_params(
    trsfm=model,
    in_spec=in_spec,
    out_spec=out_spec,
    max_in_out_seq_size=1_000_000,
)[0]

# Perform streaming inference
stream = SlidingWindowStream(model, sli_params, in_spec, out_spec)
for audio_chunk in stream.forward_in_chunks_iter(mel, chunk_size=80):
    print(f"Got a {tuple(audio_chunk.shapes[0])} shaped audio chunk")
```

## Disclaimer

TorchStream is developed by myself. It is not affiliated with, endorsed by, or sponsored by the PyTorch team or Meta.
