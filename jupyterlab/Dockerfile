#---
# name: jupyterlab
# group: core
# depends: [python, numpy, rust]
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# https://stackoverflow.com/questions/61553491/how-to-make-jupyter-terminal-fully-functional
ENV SHELL /bin/bash
RUN apt-get purge -y '*opencv*' && \
	apt update && \
    apt remove -y libopencv && \
	apt install -y vim wget alsa-base \
        alsa-utils \
        libsndfile1-dev \
		libopenblas-dev

ENV CONDA_DIR /opt/miniconda
RUN wget -nv \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
	-O /tmp/miniconda.sh \
	&& bash /tmp/miniconda.sh -b -p /opt/miniconda \
	&& rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment.yaml .
COPY environment-3.8.yaml .
RUN conda env create -f environment.yaml
RUN conda env create -f environment-3.8.yaml
# Make RUN commands use the new environment:
RUN echo "conda activate ai" >> ~/.bashrc
#ENV CMAKE_ARGS -DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=72;87
#RUN /bin/bash -c "CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS=\"${CMAKE_ARGS}\" FORCE_CMAKE=1 \
#    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade"
# enable colors in the jupyterlab terminal
#COPY llama_cpp_python-0.2.57-cp310-cp310-manylinux_2_31_aarch64.whl .
#RUN /bin/bash -c "pip3 install llama_cpp_python-0.2.57-cp310-cp310-manylinux_2_31_aarch64.whl --no-cache-dir --force-reinstall --upgrade"

RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
RUN curl -sSL https://sdk.cloud.google.com | bash
RUN pip install jupyterlab huggingface_hub[hf_transfer]
RUN conda run -n ai python -m ipykernel install --user --name=ai
RUN conda run -n ai-3.8 python -m ipykernel install --user --name=ai-3.8
RUN conda run -n ai-3.8 pip install --no-cache https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
WORKDIR /root/nb
EXPOSE 8888
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV PATH $PATH:/root/google-cloud-sdk/bin
CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --NotebookApp.token='' --NotebookApp.password='' --no-browser --allow-root --ServerApp.terminado_settings=\"shell_command=['/bin/bash']\""
