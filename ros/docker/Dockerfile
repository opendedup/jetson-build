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
	libsndfile1-dev libopenblas-dev libportaudio2 portaudio19-dev zstd python3-pip

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
	pip install -r requirements.txt && \
	rm requirements.txt && \
	python -m spacy download en_core_web_sm

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN pip install huggingface_hub[hf_transfer]
RUN pip install https://developer.download.nvidia.cn/compute/redist/jp/v60dp/pytorch/torch-2.3.0a0+6ddf5cf85e.nv24.04.14026654-cp310-cp310-linux_aarch64.whl
COPY pkgs/faiss-1.7.4-py3-none-any.whl .
RUN pip install faiss-1.7.4-py3-none-any.whl
ENV PATH $PATH:/root/google-cloud-sdk/bin
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]
