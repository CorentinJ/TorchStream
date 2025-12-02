import torch

from examples.resources.bigvgan.bigvgan import BigVGAN
from examples.resources.bigvgan.meldataset import get_mel_spectrogram
from examples.utils.audio import load_audio
from examples.utils.download import download_file_cached

device = "cuda"


def load_bigvgan() -> BigVGAN:
    model = BigVGAN.from_pretrained("nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False)
    model.remove_weight_norm()
    model = model.eval().to(device)
    return model


model = load_bigvgan()

# Load an audio file at the model's samplerate
MP3_URL = "https://d38nvwmjovqyq6.cloudfront.net/va90web25003/companions/ws_smith/32%20Speaking%20The%20Text%20As%20A%20Dramatic%20Reading.mp3"
local_audio_path = download_file_cached(MP3_URL)
wave, sample_rate = load_audio(local_audio_path, sample_rate=model.h.sampling_rate)

# Compute the mel spectrogam input
mel = get_mel_spectrogram(torch.from_numpy(wave).unsqueeze(0), model.h).to(device)

with torch.inference_mode():
    # mel is a (1, M, T_frames) shaped float32 tensor
    # wav_out is a (1, 1, T_samples) shaped float32 tensor
    wav_out = model(mel)

from torchstream import SeqSpec

# Mel spectrogram input
in_spec = SeqSpec(1, model.h.num_mels, -1, device=device)
# Audio waveform output
out_spec = SeqSpec(1, 1, -1, device=device)

# from torchstream import find_sliding_window_params

# sli_params = find_sliding_window_params(
#     model,
#     in_spec,
#     out_spec,
#     # BigVGAN produces outputs in the audio domain with a large receptive field,
#     # so the solver reaches the limit 100,000 on the input/output size while
#     # searching for a solution. We can safely increase it tenfold here.
#     max_in_out_seq_size=1_000_000,
# )[0]


from torchstream import SlidingWindowParams

sli_params = SlidingWindowParams(
    kernel_size_in=75,
    stride_in=1,
    left_pad=37,
    right_pad=37,
    kernel_size_out=314,
    stride_out=256,
    left_out_trim=29,
    right_out_trim=29,
)

print(
    f"-> min/max overlap: {[sli_params.streaming_context_size, sli_params.streaming_context_size + sli_params.stride_in - 1]}\n"
    + f"-> min/max output delay: {list(sli_params.output_delay_bounds)}\n"
    + f"-> in/out size relation: {sli_params.in_out_size_rel_repr}"
)


from torchstream import SlidingWindowStream

stream = SlidingWindowStream(model, sli_params, in_spec, out_spec)
for chunk in stream.forward_in_chunks_iter(mel, chunk_size=120):
    print(chunk.shapes)
