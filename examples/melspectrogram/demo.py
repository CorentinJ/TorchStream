# import inspect
# import logging

# import streamlit as st
# import torchaudio

# from torchstream import (
#     SeqSpec,
#     SlidingWindowStream,
#     find_sliding_window_params,
#     test_stream_equivalent,
# )

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO)


# def run_solver(n_fft: int, center: bool, pad_mode: str, hop_length: int):
#     transform = torchaudio.transforms.Spectrogram(
#         n_fft=n_fft,
#         center=center,
#         pad_mode=pad_mode,
#         hop_length=(hop_length or None),
#     )

#     in_spec = SeqSpec(-1)
#     out_spec = SeqSpec(n_fft // 2 + 1, -1)
#     solutions = find_sliding_window_params(transform, in_spec, out_spec)
#     params = solutions[0]

#     test_stream_equivalent(
#         transform,
#         SlidingWindowStream(transform, params, in_spec, out_spec),
#         in_data=in_spec.new_randn_arrays(200),
#     )


# def app() -> None:
#     st.set_page_config(page_title="Melspectrogram Sliding Window", layout="wide")
#     st.title("Mel Spectrogram Sliding-Window Solver")
#     st.write(
#         "Use the controls below to explore how different spectrogram settings impact "
#         "the sliding-window parameters inferred by `torchstream`."
#     )

#     code_col, ui_col, plot_col = st.columns([1, 1, 1], gap="medium", border=True)

#     with code_col:
#         st.subheader("Source code")
#         st.code(inspect.getsource(run_solver), language="python")

#     with ui_col:
#         n_fft = st.slider(
#             "n_fft",
#             min_value=16,
#             max_value=400,
#             value=20,
#             step=2,
#             help="Window size used by torchaudio.transforms.Spectrogram.",
#         )
#         center = st.toggle(
#             "center",
#             value=True,
#             help="When on, the audio tensor is padded so that the t-th frame is centered.",
#         )
#         pad_mode = st.radio(
#             "pad_mode",
#             options=("reflect", "constant", "replicate"),
#             index=0,
#             horizontal=True,
#             help="Padding mode that torchaudio uses when center=True.",
#         )
#         hop_length = st.slider(
#             "hop_length",
#             min_value=0,
#             max_value=n_fft,
#             value=0,
#             help="Stride between windows.",
#         )

#         with st.spinner("Finding sliding-window parameters..."):
#             run_solver(
#                 n_fft=n_fft,
#                 center=center,
#                 pad_mode=pad_mode,
#                 hop_length=hop_length,
#             )

#         st.success("Solver finished and `test_stream_equivalent` passed.")


# if __name__ == "__main__":
#     app()


import numpy as np
import streamlit as st
import torchaudio

from dev_tools.streamlit_radio_stream import streamlit_ensure_web_radio_stream
from torchstream.sequence.sequence import Sequence

radio_url = st.text_input(
    "Radio stream URL",
    value="https://icecast.radiofrance.fr/fip-midfi.mp3",
    help="Any ffmpeg-compatible stream (Icecast/HTTP).",
)

audio_stream = streamlit_ensure_web_radio_stream(radio_url)

st.sidebar.write(f"Decoded at {audio_stream.sample_rate} Hz")
st.audio(radio_url)

transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)
n_freq = transform.n_fft // 2 + 1
n_timesteps = 700
image_buff = Sequence(np.zeros((n_freq, n_timesteps)), seq_dim=1)
placeholder = st.empty()
st.caption("Live spectrogram (frequency bins x frames).")
placeholder.image(image_buff.data[0])

plot_max_value = 1e-6

while True:
    chunk = audio_stream.read()

    spec = transform(chunk).numpy()
    plot_max_value = max(plot_max_value, float(spec.max()))
    image_buff.feed(spec)
    image_buff.drop_to(n_timesteps)

    normalized = image_buff.data[0] / plot_max_value
    placeholder.image(normalized)
