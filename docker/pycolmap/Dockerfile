FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=non-interactive

#  -------------- requirements ---------------
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    vim \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libcgal-qt5-dev\
    libatlas-base-dev \
    libsuitesparse-dev \
    libflann-dev \ 
    libsqlite3-dev \
    libmetis-dev

# ---------------- ceres --------------------

RUN git clone https://ceres-solver.googlesource.com/ceres-solver \
    /software/ceres && \
    cd /software/ceres && \
    git checkout $(git describe --tags) && \
    mkdir -p /software/ceres/build && cd /software/ceres/build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DCMAKE_CUDA_ARCHITECTURES="61" && \
    make -j"$(nproc)" && \
    make install  && \
    make clean


# ------------------ colmap ---------------

# originally checkout 3.9.1 but that contains file with missing memory include and is not patched 
RUN git clone https://github.com/colmap/colmap.git \
    /software/colmap && \
    cd /software/colmap && \
    git checkout c0d8926 && \
    mkdir build && \
    cd build && \
    cmake ..  -DCMAKE_CUDA_ARCHITECTURES="61" && \
    make -j"$(nproc)" && \
    make install  && \
    make clean


# ------------ pycolmap --------------
RUN apt-get -y install python3-pip
RUN cd /software/colmap/pycolmap/ && python3 -m pip install .

# ----------- jupyter ---------------
RUN python3 -m pip install jupyter

# ------------ finish ----------------
RUN apt-get purge -y cmake && apt-get autoremove -y
RUN mkdir /workspace
WORKDIR /


# ---cuda --- 
# This should be handled by container toolkit ( and cdi support for podman)
#RUN apt-get update 
#RUN apt-get install -y wget
#RUN apt-key del 7fa2af80 && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb &&  yes | dpkg -i cuda-keyring_1.1-1_all.deb
##RUN apt-get update && apt-get -y install cuda-toolkit && apt-get -y install nvidia-gds
#RUN apt-get update && \
#    apt-get install -y software-properties-common && \
#    rm -rf /var/lib/apt/lists/*
#RUN add-apt-repository ppa:graphics-drivers/ppa
#RUN apt-get install -y nvidia-driver cuda-toolkit cuda-drivers cuda
