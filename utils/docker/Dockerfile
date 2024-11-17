
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf https://ultralytics.com/assets/Arial.Unicode.ttf /root/.config/Ultralytics/


ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Security updates
RUN apt upgrade --no-install-recommends -y openssl

# Create working directory
RUN rm -rf /usr/src/app && mkdir -p /usr/src/app
WORKDIR /usr/src/app


COPY . /usr/src/app


COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel
RUN pip install --no-cache -r requirements.txt albumentations comet gsutil notebook \
    coremltools onnx onnx-simplifier onnxruntime 'openvino-dev>=2023.0'
    # tensorflow tensorflowjs \


ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype

