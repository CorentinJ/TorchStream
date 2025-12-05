FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/root/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

COPY torchstream ./torchstream
COPY examples ./examples

RUN uv sync --group demos --python 3.11

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONPATH="/app:${PYTHONPATH}" \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8004

CMD ["streamlit", "run", "examples", "--server.port=8004", "--server.address=0.0.0.0"]
# streamlit run examples --server.port 8004
