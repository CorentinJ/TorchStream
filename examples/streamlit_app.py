from pathlib import Path

import streamlit as st

PAGES = {
    "Core examples": [
        st.Page(
            "1_introduction_with_spectrograms.py",
            title="1. Introduction to TorchStream",
            default=True,
        ),
        st.Page(
            "2_sliding_window_solver_with_resamplers.py",
            title="2. The Sliding Window Parameters Solver",
        ),
        st.Page(
            "3_streaming_bigvgan.py",
            title="3. Streaming BigVGAN",
        ),
        st.Page(
            "4_streaming_kokoro_tts.py",
            title="4. Streaming Kokoro TTS",
        ),
    ]
}


def render_prev_next(python_filepath: str | Path) -> None:
    python_filepath = Path(python_filepath)

    # FIXME: adhoc
    pages = PAGES["Core examples"]
    curr_page_idx = next(i for i, page in enumerate(pages) if page._page == python_filepath)

    prev_page = pages[curr_page_idx - 1] if curr_page_idx > 0 else None
    next_page = pages[curr_page_idx + 1] if curr_page_idx < len(pages) - 1 else None

    col_prev, col_next = st.columns(2)
    if prev_page:
        with col_prev:
            st.page_link(
                prev_page,
                label=f"⬅ {prev_page.title}",
            )
    if next_page:
        with col_next:
            st.page_link(
                next_page,
                label=f"{next_page.title} ➡",
            )


if __name__ == "__main__":
    pg = st.navigation(
        PAGES,
        expanded=True,
    )
    pg.run()
