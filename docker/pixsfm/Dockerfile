FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

ARG COLMAP_VERSION=3.8
ARG CUDA_ARCHITECTURES=86

# Prevent stop building ubuntu at time zone selection.  
ENV DEBIAN_FRONTEND=noninteractive

# Prepare and empty machine for building.
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev

# GCC needs to be >= 6.1
RUN gcc --version
RUN apt-get install -y python3 python3-pip

# ceres 2.1.0
RUN git clone https://ceres-solver.googlesource.com/ceres-solver \
    /software/ceres && \
    cd /software/ceres && \
    git checkout 2.1.0 && \
    mkdir -p /software/ceres/build && cd /software/ceres/build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES="86" && \
    make -j"$(nproc)" && \
    make install  && \
    make clean

# colmap 3.8.
RUN git clone https://github.com/colmap/colmap.git
RUN cd colmap && \
    git reset --hard ${COLMAP_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} && \
    ninja && \
    ninja install && \
    cd .. && rm -rf colmap

# pyceres v1.0
RUN git clone --recurse-submodules -b v1.0 https://github.com/cvg/pyceres.git /software/pyceres && cd /software/pyceres/  && python3 -m pip install .

# pycolmap v0.4.0
RUN git clone --recursive -b v0.4.0 https://github.com/colmap/pycolmap.git /software/pycolmap && cd /software/pycolmap && python3 -m pip install .

# hloc v1.4
RUN git clone -b v1.4 --recursive https://github.com/cvg/Hierarchical-Localization/ /software/hloc && \
    cd /software/hloc \
    python3 -m pip install -e .

# libhdf5
RUN  apt-get install -y \
    libhdf5-dev

# pixsfm
RUN git clone -b main --recurse-submodules https://github.com/cvg/pixel-perfect-sfm.git /software/pixsfm && \
    cd /software/pixsfm && \
    python3 -m pip install torch>=1.7 torchvision>=0.8 numpy tqdm>=4.36.0 matplotlib plotly scipy h5py omegaconf packaging

RUN cd /software/pixsfm && python3 -m pip install .

# ------------ finish ----------------
RUN apt-get purge -y cmake && apt-get autoremove -y
RUN mkdir -p /workspace
WORKDIR /
