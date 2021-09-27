FROM continuumio/anaconda3

RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN apt-get update && apt-get install -y gcc g++ libgl1
COPY . src/CaImAn
RUN cd src/CaImAn && conda env create -n caiman -f environment.yml && conda install --override-channels -c conda-forge -n caiman pip
RUN /bin/bash -c "cd src/CaImAn && source activate caiman && /opt/conda/envs/caiman/bin/pip install ."
RUN /bin/bash -c "source activate caiman && caimanmanager.py install"

