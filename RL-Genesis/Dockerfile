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
  pip install libigl==2.5.1 genesis-world rsl-rl-lib tensorboard optuna

WORKDIR /src

COPY . /src

RUN cp ./rl_rsl_ppo.py /usr/local/lib/python3.10/dist-packages/rsl_rl/algorithms/ppo.py