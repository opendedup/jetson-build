#---
# name: jupyterlab
# group: core
# depends: [python, numpy, rust]
# notes: will autostart Jupyter server on port 8888 unless container entry CMD is overridden
#---
ARG BASE_IMAGE
ARG CUDAARCHS
FROM ${BASE_IMAGE}

# https://stackoverflow.com/questions/61553491/how-to-make-jupyter-terminal-fully-functional
ENV SHELL /bin/bash
SHELL ["/bin/bash", "-c"]
RUN apt update && \
	apt upgrade -y && \
	apt install -y vim wget

ENV CONDA_DIR /opt/miniconda
RUN wget -nv \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh \
	-O /tmp/miniconda.sh \
	&& bash /tmp/miniconda.sh -b -p /opt/miniconda \
	&& rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
COPY environment.yaml .
RUN conda env create -f environment.yaml

# Make RUN commands use the new environment:
RUN echo "conda activate ai" >> ~/.bashrc

RUN CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUBLAS=on -DCMAKE_CUDA_ARCHITECTURES=${CUDAARCHS}" FORCE_CMAKE=1 \
    pip install jupyterlab llama-cpp-python --no-cache-dir --force-reinstall --upgrade
# enable colors in the jupyterlab terminal
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /root/.bashrc
WORKDIR /root
CMD /bin/bash -c "jupyter lab --ip 0.0.0.0 --port 8888 --allow-root"
