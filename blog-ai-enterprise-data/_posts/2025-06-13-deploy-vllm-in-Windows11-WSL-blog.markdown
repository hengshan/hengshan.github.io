---
layout: post-wide
title:  "Dual-System Deep Learning: Deploying vLLM with WSL2 and Docker on Windows/Linux Hybrid Setup"
date:  2025-06-10 22:42:32 +0800
category: AI 
author: Hank Li
---

## Introduction

In my quest to build a powerful RAG (Retrieval-Augmented Generation) LLM application environment, I've created a hybrid setup combining:
- A new ROG STRIX 5090 laptop (24GB VRAM) running Windows 11
- My old Lenovo Legion 7i (RTX 3060 6GB VRAM) repurposed as a Linux server (Ubuntu 24.04 LTS)

This post documents the challenges and solutions for setting up vLLM across these systems, with particular focus on WSL2 configuration, Docker GPU integration, and cross-system service exposure.

## Overcoming Installation Hurdles

### Failed Attempt: Native Linux Installation on ROG STRIX

Initially, I tried installing Linux natively on the ROG STRIX 5090 using Ventoy USB, but encountered BIOS limitations:
- BIOS didn't recognize the boot disk
- Intel Disk Management setting was not allowed to change

**Help wanted:** If anyone knows how to modify these BIOS settings of ROG STRIX for dual-boot, please share in the comments!

### Successful Alternative: WSL2 with GPU Acceleration
Since native installation failed, I opted for WSL2 with full GPU support:
1. **Windows-side prerequisites**:
   ```powershell
   # Install NVIDIA driver for Windows
   # Download from NVIDIA website matching your GPU model
   
   # Install CUDA Toolkit for Windows
   ```

2. **WSL2 setup**:
   ```bash
   # Enable WSL and install Ubuntu 24.04 LTS
    wsl --install -d Ubuntu-24.04

    # Inside WSL:
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y build-essential python3-dev
   ```

3. **WSL2 GPU**:
   ```bash
    # Install NVIDIA CUDA toolkit in WSL
    wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt update
    sudo apt install -y cuda-toolkit-12-9
   ```

4. **Docker and NVIDIA Container Toolkit**
For GPU-accelerated containers in WSL2:
   ```bash
    # Install Docker in WSL
    sudo apt install -y docker.io

    # Add NVIDIA container toolkit
    distribution=$(. /etc/os-release;echo $UBUNTU_CODENAME) \
       && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
       && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
       
    sudo apt update
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
   ```
5. **Install VLLM using pip or uv**

I experimented with both uv and conda environments. Using UV is faster when install packages. However, UV does not have pip dev libararies, which are necessary when install vllm locally. You can use sudo apt update && sudo apt install python3-dev, or can simply use conda env, which has all the dev libararies. 
```bash
    # Using uv (ultra-fast pip alternative)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.cargo/env
# Create environment
    uv venv --python 3.12 .venv --seed
    source .venv/bin/activate
# Install system dependencies
    sudo apt install -y python3-dev ccache
# install pytorch
    uv pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    # install requirmentes
    python use_existing_torch.py
    uv pip install -r requirements/build.txt
# Install with ccache optimizations
    CCACHE_NOHASHDIR="true" uv pip install --no-build-isolation -e .
   ```
Note: if you use a relatively new GPU, such as 5090 RTX as I use. VLLM may not have the wheel ready. You need to manually install pytorch and then.

## Serving Models with vLLM
Primary LLM Service (DeepSeek-R1-7B) is a deepseek distilled QWEN model. 
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./DeepSeek-R1-8B \
    --served-model-name deepseek-r1-8b \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 20000 \
    --trust-remote-code
```
Note: On the 5090 with 24GB VRAM, this leaves 1-2GB free.

## Cross-System Networking Challenges
### The vllm service is running within WSL in Windows.
Connecting Docker Containers that also run within WLS to VLLM Services. For example, when you deploy Dify or RagFlow in WSL docker, you can input http://172.17.0.1:8000/v1 as the API url. Do not use wsl ip, as it may change after reboot. In my laptop, I need to use 172.17.0.1 to connect successfully, as it is Docker wsl gateway.


### Exposing WSL Services to LAN.
To make vLLm accessible to other machines (like the Linux server):

1. **Port forwarding on Windows:**
```powershell
# Forward port 8000 to WSL (run as Admin)
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=172.28.139.149
# Verify rules
netsh interface portproxy show all
```
2. **Firewall configuration:**
```powershell
netsh advfirewall firewall add rule name="vLLM Port 8000" dir=in action=allow protocol=TCP localport=8000
```
3. **Remote configuration:**
```bash
# On Linux server accessing the Windows host
"vllm_base_url": "http://192.168.10.3:8000/v1"
```


## Conclusion
This hybrid setup proves that with proper configuration:

- WSL2 can serve as a viable Linux development environment for GPU workloads
- Cross-system model serving is possible with careful networking setup
- Modern Windows/Linux interoperability has reached a practical level for ML workloads

The entire system now serves as a powerful platform for developing and testing RAG applications with multiple specialized models working in concert.
