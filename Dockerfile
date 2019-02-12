FROM continuumio/anaconda3

RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN mkdir src && cd src && git clone -b dev https://github.com/flatironinstitute/CaImAn.git && cd CaImAn && conda env create -n caiman -f environment.yml && conda activate caiman && pip install .
RUN conda activate caiman && caimanmanager.py install

