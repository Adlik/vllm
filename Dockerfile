# The vLLM Dockerfile is used to construct vLLM image that can be directly used
# to run the OpenAI compatible server.

# Please update any changes made here to
# docs/source/dev/dockerfile/dockerfile.rst and
# docs/source/assets/dev/dockerfile-stages-dependency.png

ARG CUDA_VERSION=12.1.0
#################### BASE BUILD IMAGE ####################
# prepare basic build environment
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS base
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/10no--check-valid-until \
    && echo 'Acquire::AllowInsecureRepositories "true";' >> /etc/apt/apt.conf.d/10no--check-valid-until

# Install Python and other dependencies

COPY get-pip.py /get-pip.py

RUN apt-get install -y tzdata \
    && echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/Asia select Shanghai' | debconf-set-selections \
    && apt-get install -y ccache git curl sudo \
    && apt-get update -y \
    && apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config

RUN python3 /get-pip.py \
    && python3 --version && python3 -m pip --version

# Upgrade to GCC 10 to avoid https://gcc.gnu.org/bugzilla/show_bug.cgi?id=92519
# as it was causing spam when compiling the CUTLASS kernels
RUN apt-get install -y gcc-10 g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 --slave /usr/bin/g++ g++ /usr/bin/g++-10
RUN gcc --version

# Workaround for https://github.com/openai/triton/issues/2507 and
# https://github.com/pytorch/pytorch/issues/107960 -- hopefully
# this won't be needed for future versions of this docker image
# or future versions of triton.
RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

WORKDIR /workspace

# install build and runtime dependencies
COPY requirements-common.txt requirements-common.txt
COPY requirements-cuda.txt requirements-cuda.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-cuda.txt


# cuda arch list used by torch
# can be useful for both `dev` and `test`
# explicitly set the list to avoid issues with torch 2.2
# see https://github.com/pytorch/pytorch/pull/123243
ARG torch_cuda_arch_list='7.0 7.5 8.0 8.6 8.9 9.0+PTX'
ENV TORCH_CUDA_ARCH_LIST=${torch_cuda_arch_list}
# Override the arch list for flash-attn to reduce the binary size
ARG vllm_fa_cmake_gpu_arches='80-real;90-real'
ENV VLLM_FA_CMAKE_GPU_ARCHES=${vllm_fa_cmake_gpu_arches}
#################### BASE BUILD IMAGE ####################

#################### WHEEL BUILD IMAGE ####################
FROM base AS build

# install build dependencies
COPY requirements-build.txt requirements-build.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-build.txt

COPY . .

# max jobs used by Ninja to build extensions
ARG max_jobs=4
ENV MAX_JOBS=${max_jobs}
# number of threads used by nvcc
ARG nvcc_threads=32
ENV NVCC_THREADS=$nvcc_threads

ARG USE_SCCACHE
ARG SCCACHE_BUCKET_NAME=vllm-build-sccache
ARG SCCACHE_REGION_NAME=cn-north-1
ARG SCCACHE_S3_NO_CREDENTIALS=0
# if USE_SCCACHE is set, use sccache to speed up compilation
COPY sccache.tar.gz sccache.tar.gz
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" = "1" ]; then \
        echo "Installing sccache..." \
        && tar -xzf sccache.tar.gz \
        && sudo mv sccache-v0.8.1-x86_64-unknown-linux-musl/sccache /usr/bin/sccache \
        && rm -rf sccache.tar.gz sccache-v0.8.1-x86_64-unknown-linux-musl \
        && export SCCACHE_BUCKET=${SCCACHE_BUCKET_NAME} \
        && export SCCACHE_REGION=${SCCACHE_REGION_NAME} \
        && export SCCACHE_S3_NO_CREDENTIALS=${SCCACHE_S3_NO_CREDENTIALS} \
        && export SCCACHE_IDLE_TIMEOUT=0 \
        && export CMAKE_BUILD_TYPE=Release \
        && sccache --show-stats \
        && SETUPTOOLS_SCM_PRETEND_VERSION="0.6.3.post1" python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38 \
        && sccache --show-stats; \
    fi

ENV CCACHE_DIR=/root/.cache/ccache

RUN mkdir -p /root/cutlass

COPY cutlass/ /root/cutlass


RUN mkdir -p /root/flash-attention

COPY flash-attention/ /root/flash-attention


RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_SCCACHE" != "1" ]; then \
        SETUPTOOLS_SCM_PRETEND_VERSION="0.6.3.post1" python3 setup.py bdist_wheel --dist-dir=dist --py-limited-api=cp38; \
    fi

# Check the size of the wheel if RUN_WHEEL_CHECK is true
COPY .buildkite/check-wheel-size.py check-wheel-size.py
# Default max size of the wheel is 250MB
ARG VLLM_MAX_SIZE_MB=400
ENV VLLM_MAX_SIZE_MB=$VLLM_MAX_SIZE_MB
ARG RUN_WHEEL_CHECK=true
RUN if [ "$RUN_WHEEL_CHECK" = "true" ]; then \
        python3 check-wheel-size.py dist; \
    else \
        echo "Skipping wheel size check."; \
    fi
#################### EXTENSION Build IMAGE ####################

#################### DEV IMAGE ####################
FROM base as dev

COPY requirements-lint.txt requirements-lint.txt
COPY requirements-test.txt requirements-test.txt
COPY requirements-dev.txt requirements-dev.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements-dev.txt

#################### DEV IMAGE ####################
#################### vLLM installation IMAGE ####################
# image with vLLM installed
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04 AS vllm-base
ARG CUDA_VERSION=12.1.0
ARG PYTHON_VERSION=3.10
WORKDIR /vllm-workspace
ENV DEBIAN_FRONTEND=noninteractive

RUN PYTHON_VERSION_STR=$(echo ${PYTHON_VERSION} | sed 's/\.//g') && \
    echo "export PYTHON_VERSION_STR=${PYTHON_VERSION_STR}" >> /etc/environment

RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo 'Acquire::Check-Valid-Until "false";' >> /etc/apt/apt.conf.d/10no--check-valid-until \
    && echo 'Acquire::AllowInsecureRepositories "true";' >> /etc/apt/apt.conf.d/10no--check-valid-until

RUN apt-get install -y tzdata \
    && echo 'tzdata tzdata/Areas select Asia' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/Asia select Shanghai' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y ccache  git curl sudo vim python3-pip \
    && apt-get install -y ffmpeg libsm6 libxext6 libgl1

COPY get-pip.py /get-pip.py

RUN apt-get update -y && apt-cache search python3 \
    && apt-get install -y python${PYTHON_VERSION}  python${PYTHON_VERSION}-dev python${PYTHON_VERSION}-venv libibverbs-dev \
    && which python${PYTHON_VERSION} \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} \
    && ls /usr/bin/python* \
    && ln -sf /usr/bin/python${PYTHON_VERSION}-config /usr/bin/python3-config


RUN python3 /get-pip.py \
    && python3 --version && python3 -m pip --version


RUN ldconfig /usr/local/cuda-$(echo $CUDA_VERSION | cut -d. -f1,2)/compat/

# install vllm wheel first, so that torch etc will be installed
RUN --mount=type=bind,from=build,src=/workspace/dist,target=/vllm-workspace/dist \
    --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install dist/*.whl --verbose

RUN --mount=type=cache,target=/root/.cache/pip \
    . /etc/environment
COPY examples examples

FROM vllm-base AS vllm-openai

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install accelerate hf_transfer 'modelscope!=1.15.0' bitsandbytes>=0.44.0 timm==0.9.10

ENV VLLM_USAGE_SOURCE production-docker-image

