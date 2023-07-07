FROM continuumio/anaconda3

RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN apt-get update && apt-get install -y gcc g++ libgl1
RUN conda create -n caiman -c conda-forge caiman
RUN /bin/bash -c "source activate caiman && caimanmanager install"

