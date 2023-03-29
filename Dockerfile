# Copyright (c) 2023, The Wordcab team. All rights reserved.
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

COPY requirements.txt /requirements.txt
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libsndfile1 \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.10 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3.10 -m pip install -r requirements.txt \
    && python3.10 -m pip install --upgrade torch==1.13.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

COPY . /app
WORKDIR /app

CMD ["uvicorn", "--reload", "--host=0.0.0.0", "--port=5001", "asr_api.main:app"]