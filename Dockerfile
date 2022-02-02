FROM nvcr.io/nvidia/pytorch:21.12-py3

WORKDIR /home/jovyan/work 

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    # python
    python3.8 python3-pip python3-setuptools python3-dev

RUN rm -rf /var/cache/apt/* /var/lib/apt/lists/* && \
    apt-get autoremove -y && apt-get clean

RUN pip install clara-viz clara-viz-core clara-viz-widgets

FROM jupyter/minimal-notebook

RUN git clone https://github.com/NVIDIA/clara-viz

