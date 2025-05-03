FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set noninteractive mode
ENV DEBIAN_FRONTEND=noninteractive

# Install system packages
RUN apt-get update && apt-get install -y \
    git wget sudo nano curl unzip build-essential cmake ninja-build \
    python3 python3-pip python3-dev python-is-python3 \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgl1-mesa-glx libgl1-mesa-dri ffmpeg \ 
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install PyTorch 2.0.1 (CUDA 11.8)
RUN pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Add user dmaji and set permissions
RUN useradd -m -s /bin/bash dmaji && \
    echo "dmaji ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    chown -R dmaji: /home/dmaji

# Add bashrc
COPY ./.devcontainer/.bashrc /home/dmaji/.bashrc
RUN chown dmaji: /home/dmaji/.bashrc

# Switch to user dmaji
USER dmaji

# Set working directory
WORKDIR /home/dmaji

CMD ["/bin/bash"]
