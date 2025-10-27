# from diffusers import StableAudioPipeline
# import torch

# pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16).to("cuda")
# audio = pipe(prompt="cinematic drum loop, 120 bpm, clean, punchy", audio_length_in_s=20).audios[0]


import torch

print(torch.cuda.is_available())
