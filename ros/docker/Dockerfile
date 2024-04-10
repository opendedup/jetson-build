#---
# name: jupyterlab
# group: core
# depends: [python, numpy, rust]
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
FROM dustynv/ros:humble-ros-core-l4t-r35.4.1

# https://stackoverflow.com/questions/61553491/how-to-make-jupyter-terminal-fully-functional
ENV SHELL /bin/bash
RUN apt-get purge -y '*opencv*' && \
	apt update && \
    apt remove -y libopencv && \
	apt install -y vim wget alsa-base \
        alsa-utils \
        libsndfile1-dev \
		libopenblas-dev libportaudio2 portaudio19-dev zstd

COPY requirements.txt .
RUN python3 -m pip install --upgrade pip && \
	pip install -r requirements.txt && \
	rm requirements.txt && \
	python -m spacy download en_core_web_sm

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN pip install huggingface_hub[hf_transfer]
RUN pip install https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
COPY pkgs/faiss-1.7.4-py3-none-any.whl .
RUN pip install faiss-1.7.4-py3-none-any.whl
ENV PATH $PATH:/root/google-cloud-sdk/bin
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["/bin/bash"]
