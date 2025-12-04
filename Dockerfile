FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

COPY pyproject.toml uv.lock README.md LICENSE ./

COPY torchstream ./torchstream
COPY examples ./examples

RUN uv sync --group demos

ENV PATH="/app/.venv/bin:${PATH}" 

EXPOSE 8004

CMD ["streamlit", "run", "examples/1_introduction_with_spectrograms.py", "--server.port=8004", "--server.address=0.0.0.0"]
