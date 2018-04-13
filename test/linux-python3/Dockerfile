FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y \
    gcc g++ \
    libgtk2.0-0 \
    libc6-i386 \
    libgl1-mesa-swx11 \
    xvfb \
  && rm -rf /var/lib/apt/lists/*
# RUN conda install -c menpo opencv3=3.1.0
# RUN conda install -c cvxgrp cvxpy
# RUN conda install -c https://conda.anaconda.org/conda-forge tifffile
# RUN git clone --recursive -b agiovann-master https://github.com/valentina-s/Constrained_NMF.git
# RUN git clone --recursive https://github.com/agiovann/Constrained_NMF.git
# RUN git clone --recursive -b dev https://github.com/agiovann/Constrained_NMF.git
