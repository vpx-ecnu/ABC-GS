FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update -y && apt-get install -y --no-install-recommends \
    wget \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/vpx-ecnu/ABC-GS --recursive

RUN pip install \
    gs/submodules/diff-gaussian-rasterization \
    gs/submodules/simple-knn \
    abcgs/submodules/lang-segment-anything \
    imageio \
    wandb \
    plyfile \
    open3d \
    simple-parsing \
    opencv-python \
    icecream

CMD ["/bin/bash"]