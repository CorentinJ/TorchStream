import librosa
import soundfile as sf
import torch

from examples.resources.bigvgan import bigvgan
from examples.resources.bigvgan.meldataset import get_mel_spectrogram

device = "cuda"

# instantiate the model. You can optionally set use_cuda_kernel=True for faster inference.
model = bigvgan.BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)

# remove weight norm in the model and set to eval mode
model.remove_weight_norm()
model = model.eval().to(device)

# load wav file and compute mel spectrogram
wav_path = r"C:\Users\coren\Downloads\32 Speaking The Text As A Dramatic Reading.mp3"
wav, sr = librosa.load(
    wav_path, sr=model.h.sampling_rate, mono=True
)  # wav is np.ndarray with shape [T_time] and values in [-1, 1]
wav = torch.FloatTensor(wav).unsqueeze(0)  # wav is FloatTensor with shape [B(1), T_time]

# compute mel spectrogram from the ground truth audio
mel = get_mel_spectrogram(wav, model.h).to(device)  # mel is FloatTensor with shape [B(1), C_mel, T_frame]

# generate waveform from mel
with torch.inference_mode():
    wav_gen = model(mel)  # wav_gen is FloatTensor with shape [B(1), 1, T_time] and values in [-1, 1]
wav_gen_float = wav_gen.cpu().flatten()  # wav_gen is FloatTensor with shape [1, T_time]


sf.write("debug.wav", wav_gen_float.numpy(), samplerate=model.h.sampling_rate)
