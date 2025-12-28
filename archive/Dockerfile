
# Tested on (GPUs, CUDA, Driver): tag
# (L4, 12.2, 535): 23.10-py3
# FROM nvcr.io/nvidia/pytorch:23.10-py3 AS base
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# Go to workspace
WORKDIR /workspace

RUN bash -c "pip uninstall numpy pyarrow pandas -y \
    && pip install opencv-fixer==0.2.5"


RUN bash -c "git clone -b n1.5-release https://github.com/NVIDIA/Isaac-GR00T.git \
    && cd Isaac-GR00T \
    && pip install -e .[base] "

RUN bash -c "pip install --no-build-isolation flash-attn==2.7.1.post4"

RUN bash -c "python -c \"from opencv_fixer import AutoFix; AutoFix()\""

# Add groot_entrypoint.sh script
# COPY agent/groot_entrypoint.sh /workspace/entrypoint.sh

# Clone the groot repository
# RUN bash -c "chmod +x /workspace/groot_entrypoint.sh"
WORKDIR /workspace/finetune-groot
