FROM nvidia/cuda:12.4.0-devel-ubuntu22.04
LABEL maintainer=""

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workdir
ENV APP_PATH=/workdir

# System dependencies
RUN apt-get update -y && \
    apt-get install -y curl git bash nano software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update -y && \
    apt-get install -y python3.11 python3.11-dev python3.11-distutils python3-pip && \
    apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --set python3 /usr/bin/python3.11

RUN python3 -m pip install --upgrade pip

# System deps for PDF processing (poppler required by pdf2image, libgl1 for opencv)
RUN apt-get update -y && \
    apt-get install -y poppler-utils libgl1 && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Project dependencies
COPY docker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# src/ is on PYTHONPATH so imports work as: from core.utils import ...
ENV PYTHONPATH=${APP_PATH}/src:$PYTHONPATH

ENV DEBIAN_FRONTEND=dialog
