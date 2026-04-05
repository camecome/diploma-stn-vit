FROM nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04

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

WORKDIR /diploma-stn-vit
RUN python3 -m venv vit-env && vit-env/bin/pip3 install torch torchvision numpy tqdm tensorboard scipy ml_collections
RUN mkdir diploma_stn_vit imagenet1k

CMD ["bash"]
