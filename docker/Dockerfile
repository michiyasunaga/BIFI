FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu16.04
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    iputils-ping \
    libhdf5-dev \
    software-properties-common \
    swig \
    unzip \
    vim \
    wget && \
    rm -rf /var/lib/apt/lists

# Install Python3 from miniconda.
RUN curl -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \
     rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda install python=3.7.7
RUN conda --version

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir torch==1.4.0 torchvision==0.5.0 && \
    python3 -m pip install --no-cache-dir \
    tqdm==4.53.0 \
    numpy==1.20.1 \
    editdistance

ADD fairseq.tar /root/
RUN cd /root/fairseq && python3 -m pip install --no-cache-dir .
