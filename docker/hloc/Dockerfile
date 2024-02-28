FROM nvidia/cuda:12.3.1-devel-ubuntu20.04
ARG PYTHON_VERSION=3.8

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common git
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-distutils
RUN wget https://bootstrap.pypa.io/get-pip.py && python${PYTHON_VERSION} get-pip.py
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1

RUN pip install --upgrade pip
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization/ && cd Hierarchical-Localization/ && python -m pip install -e .


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install jupyter
