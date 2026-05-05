FROM --platform=linux/amd64 nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Base OS deps. We need git for editable VCS installs (triton_kernels via
# pip), build-essential for any sdist that compiles C, libgl1/libglib2.0-0
# for opencv-python-headless, and curl/gnupg for the NodeSource repo.
RUN rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        wget git curl ca-certificates gnupg \
        build-essential \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Node.js 20 — the slide-gen pipeline shells out to html2pdf.mjs (puppeteer).
# Those scripts live outside this image (see SLIDE_HTML2PDF_SCRIPT below);
# we only need a `node` binary in PATH so the subprocess can launch when
# the host tree is mounted at runtime.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Use bash for all subsequent RUN steps so `source` and `conda activate` work
SHELL ["/bin/bash", "-c"]

# --- Miniconda ---
ENV CONDA_DIR=/opt/conda
RUN wget -qO /tmp/miniconda.sh \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh
ENV PATH="$CONDA_DIR/bin:$PATH"

# --- conda env: professor (Python 3.12 to match host venv312) ---
RUN conda create -y -n professor python=3.12.* pip \
        --override-channels -c conda-forge \
    && conda clean -afy

# --- PyTorch 2.10.0 (CUDA 12.8 wheels) ---
# Matches the cu12.8 nvidia-* packages frozen in requirements.txt.
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate professor \
    && python --version && which python && which pip \
    && python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0

# --- Python deps from requirements.txt ---
# Pulls vLLM 0.18, fastapi, sentence-transformers, lancedb, PyMuPDF,
# boto3, google-auth, etc. torch is already pinned + installed above so
# pip will see it as satisfied.
WORKDIR /app
COPY requirements.txt ./
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate professor \
    && python -m pip install --no-cache-dir -r requirements.txt

# --- Local gpt_oss package (vllm.token_generator etc) ---
# pyproject.toml's default build mode produces a pure Python wheel
# (GPTOSS_BUILD_METAL is unset) so no CMake/Metal toolchain is required.
COPY pyproject.toml MANIFEST.in README.md CMakeLists.txt ./
COPY _build/ ./_build/
COPY gpt_oss/ ./gpt_oss/
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate professor \
    && python -m pip install --no-cache-dir --no-build-isolation -e .

# --- Server source: HTTP server + RAG + index builder + frontend assets ---
COPY your_professor_server.py paper_rag.py build_paper_index.py ./
COPY html/ ./html/
COPY js/ ./js/

# Persistent state (sessions, uploads, RAG index) is written under
# DATA_DIR (default ./professor_data). Mount a volume here in production
# so it survives container restarts.
RUN mkdir -p /app/professor_data/sessions /app/professor_data/uploads /app/professor_data/index

# --- Slide-gen pipeline (lives outside this image) ---
# your_professor_server invokes the scripts under /home/video-dev/git/paper/
# via subprocess for the upload→slides→PDF→narration flow. Bind-mount that
# tree at runtime if you need slide regeneration; the rest of the server
# (chat, presence, QA, RAG) works without it.
ENV SLIDE_GEN_PYTHON=/opt/conda/envs/professor/bin/python \
    SLIDE_NODE=node \
    PAPER_RAG_DEVICE=cpu \
    PAPER_RAG_RERANKER_DEVICE=cpu \
    PAPER_DIR=/app/paper \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --port default in your_professor_server.py is 8081 — matches the tunnel side.
EXPOSE 8081

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "professor"]
CMD ["python", "your_professor_server.py", "--host", "0.0.0.0", "--port", "8081"]
