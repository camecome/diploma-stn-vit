FROM docker.io/nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
RUN python3 -m venv /workspace/vit-env

RUN /workspace/vit-env/bin/pip install --upgrade pip && \
    /workspace/vit-env/bin/pip install torch torchvision

ENV PATH="/workspace/vit-env/bin:$PATH"

CMD ["bash"]
