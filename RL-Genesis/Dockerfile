FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHONDONTWRITEBYTECODE=1

RUN apt-get update -y && apt-get install -y \
  libegl1 \
  libgl1 \
  libglvnd-dev \
  libgl1-mesa-glx \
  libglew-dev \
  libegl-dev \
  libx11-6  \
  libxrender1 \
  libglu1-mesa \
  libglib2.0-0 \
  libegl1-mesa-dev \
  libgles2-mesa-dev \
  libosmesa6-dev \
  mesa-utils \
  python3-pip \
  git

RUN pip install torch --index-url https://download.pytorch.org/whl/cu121 && \
  pip install genesis-world rsl-rl-lib==2.2.4 tensorboard

WORKDIR /src

COPY . /src