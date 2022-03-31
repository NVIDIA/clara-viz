FROM nvidia/cuda:11.4.2-base-ubuntu20.04
#FROM nvcr.io/nvidia/pytorch:21.12-py3

RUN apt update \
    && apt install -y --no-install-recommends --fix-missing \
    python3.8 python3-pip python3-setuptools python3-dev \
    curl \
    git

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash

WORKDIR /home/jovyan/work
RUN git clone https://github.com/NVIDIA/clara-viz

RUN python3 -m pip install jupyter && python3 -m pip install jupyterlab
RUN python3 -m pip install clara-viz clara-viz-core clara-viz-widgets
RUN python3 -m pip install itk

WORKDIR /home/jovyan/work

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]



