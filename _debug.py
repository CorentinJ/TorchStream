import streamlit as st
import torch

from examples.resources.bigvgan import bigvgan
from examples.resources.bigvgan.melspectrogram import get_mel_spectrogram
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached

device = "cuda"

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False, strict=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

# load wav file and compute mel spectrogram
MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"
local_audio_path = download_file_cached(MP3_URL)
wav, sr = load_audio(local_audio_path, sample_rate=model.h.sampling_rate)
wav = torch.FloatTensor(wav).unsqueeze(0)  # wav is FloatTensor with shape [B(1), T_time]

# compute mel spectrogram from the ground truth audio
mel = get_mel_spectrogram(wav, model.h).to(device)  # mel is FloatTensor with shape [B(1), C_mel, T_frame]

# generate waveform from mel
with torch.inference_mode():
    wav_gen = model(mel)  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
wav_gen_float = wav_gen.squeeze(0).cpu()  # wav_gen is FloatTensor with shape [1, T_time]

st.audio(wav_gen_float.numpy(), sample_rate=model.h.sampling_rate)
