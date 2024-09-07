#---
# name: jupyterlab
# group: core
# depends: [python, numpy, rust]
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
FROM dustynv/ros:humble-llm-r36.3.0

# https://stackoverflow.com/questions/61553491/how-to-make-jupyter-terminal-fully-functional
ENV SHELL /bin/bash
RUN apt update && \
	apt upgrade -y && \
	apt install -y vim wget alsa-base \
	alsa-utils \
	libsndfile1-dev libopenblas-dev libportaudio2 portaudio19-dev zstd python3-pip libhdf5-dev libopencc-dev \
	ffmpeg libavformat-dev libavcodec-dev libavutil-dev libavdevice-dev libavfilter-dev

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
	pip install --no-cache-dir -r requirements.txt && \
	rm requirements.txt && \
	python -m spacy download en_core_web_sm

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN pip install --no-cache-dir huggingface_hub[hf_transfer]
RUN pip install --no-cache-dir https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl
COPY pkgs/faiss-1.7.4-py3-none-any.whl .
RUN pip install --no-cache-dir faiss-1.7.4-py3-none-any.whl
ENV PATH $PATH:/root/google-cloud-sdk/bin

# Install Nemo
ARG SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN pip install --no-cache-dir --ignore-installed --upgrade --no-cache-dir --verbose PyYAML

RUN git clone https://github.com/google/sentencepiece && \
	cd sentencepiece && \
	git checkout 635fe84 && \
	mkdir build && \
	cd build && \
	cmake .. && \
	make -j $(nproc) && \
	make install && \
	ldconfig -v && \
	cd .. && \
	cd python && \
	python3 setup.py install --verbose && \
	cd ../../ && \
	rm -r -f sentencepiece


RUN git clone --depth=1 -b v1.23.0 --recursive https://github.com/nvidia/nemo && \
    cd nemo && \
    sed 's|^numpy.*|numpy|g' -i requirements/requirements.txt && \
    cat requirements/requirements.txt && \
    sed 's|import nemo.collections.nlp.data.text_normalization.constants as constants|from nemo.collections.nlp.data.text_normalization import constants|g' -i nemo/collections/nlp/models/duplex_text_normalization/duplex_decoder.py && \
    head -n 100 nemo/collections/nlp/models/duplex_text_normalization/duplex_decoder.py && \
    python3 setup.py --verbose bdist_wheel && \
    cp dist/nemo*.whl /opt

RUN LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    pip install --no-cache-dir --verbose opencc

RUN pip install --no-cache-dir --verbose /opt/nemo_toolkit-1.23.0-py3-none-any.whl

# # pytorch_lightning => import wandb.sdk.lib.json_util as json
# # AttributeError: module 'wandb' has no attribute 'sdk'
RUN pip install --no-cache-dir --verbose 'wandb<0.15.8' && \
	pip install --no-cache-dir hydra-core pytorch-lightning youtokentome cmake ninja editdistance lhotse webdataset jiwer IPython
WORKDIR /opt
RUN git clone https://github.com/pytorch/audio

WORKDIR /opt/audio
ARG USE_CUDA=1
RUN USE_CUDA=1 pip install --no-cache-dir -v -e . --no-use-pep517

# # make sure it loads
RUN pip show nemo_toolkit && python3 -c 'import nemo; print(nemo.__version__)'

# # set the nemo model cache directory to mounted /data volume
ENV NEMO_CACHE_DIR=/data/models/nemo
WORKDIR /opt/
RUN git clone https://github.com/pyannote/pyannote-audio.git -b 3.3.1
RUN sed -i '31s/.*/    mine = VersionInfo.parse(mine[:5])/' /opt/pyannote-audio/pyannote/audio/utils/version.py
WORKDIR /opt/pyannote
RUN pip --no-cache-dir install -e .

WORKDIR /root/ros2_ws
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]
