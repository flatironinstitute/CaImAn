FROM ubuntu
RUN apt-get install bzip2
RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y git wget
RUN export MINICONDA=$HOME/miniconda
RUN export PATH="$MINICONDA/bin:$PATH"
RUN hash -r
RUN wget -q https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda2-2.5.0-Linux-x86_64.sh -O anaconda.sh
# RUN bash anaconda.sh -b -p $ANACONDA
RUN bash anaconda.sh -p /anaconda -b
ENV PATH=/anaconda/bin:${PATH}
RUN conda config --set always_yes yes
RUN conda update --yes conda
RUN conda info -a
RUN conda install psutil
RUN conda install ipyparallel
RUN conda install scikit-image
RUN conda install cvxopt
RUN conda install -c https://conda.anaconda.org/omnia cvxpy
RUN conda install -c https://conda.anaconda.org/conda-forge tifffile
RUN git clone --recursive https://github.com/agiovann/Constrained_NMF.git
WORKDIR /Constrained_NMF/
# RUN git checkout docker
RUN apt-get install libc6-i386
RUN apt-get install -y libsm6 libxrender1
RUN conda install pyqt
RUN python setup.py install

# RUN nosetests

EXPOSE 8080
