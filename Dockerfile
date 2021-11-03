

FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
	 unzip \
	 curl \
         cmake \
         ca-certificates \
         python3-setuptools \
         python3.6-dev \
         python3-pip && \
     rm -rf /var/lib/apt/lists/*

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PATH /opt/conda/bin:/usr/local/nvidia/bin/:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN pip3 install wheel
RUN pip3 install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

RUN pip3 install --upgrade pip

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY prompts/ prompts/
COPY tasks/ tasks/
COPY data.py  .
COPY generate_k_shot_data.py  .
COPY inspect_imbalance.py  .
COPY main.py  .
COPY model_util.py  .
COPY run.py  .
COPY templates.py  .
COPY util.py  .

