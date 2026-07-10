FROM vllm/vllm-openai:v0.17.0
LABEL maintainer=""

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workdir
ENV APP_PATH=/workdir

# System deps for PDF processing (poppler required by pdf2image, libgl1 for opencv)
RUN apt-get update -y && \
    apt-get install -y poppler-utils libgl1 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Install uv for conflict-aware package resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Project dependencies
COPY docker/6000/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# src/ is on PYTHONPATH so imports work as: from core.utils import ...
ENV PYTHONPATH=${APP_PATH}/src:$PYTHONPATH

ENV DEBIAN_FRONTEND=dialog

# Reset the VLLM base image entrypoint so we can run arbitrary commands
ENTRYPOINT []
CMD ["/bin/bash"]
