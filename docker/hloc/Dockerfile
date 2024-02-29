FROM tensorflow/tensorflow:2.15.0-gpu-jupyter
ARG PYTHON_VERSION=3.8

# Prevent stop building ubuntu at time zone selection.
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y
RUN apt-get install -y unzip wget software-properties-common git

RUN pip install --upgrade pip
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization/ && cd Hierarchical-Localization/ && python -m pip install -e .


RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install jupyter
