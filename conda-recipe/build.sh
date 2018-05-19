#!/bin/bash

echo "---------------- Running build.sh ----------------"
#conda env update -f environment_mac.yml
#conda install -c conda-forge tensorflow -y
#conda install -c conda-forge keras -y
#conda install -c menpo opencv3=3.1.0 -y
#conda install -c menpo opencv3 -y

#$PYTHON setup.py build_ext -i
#$PYTHON setup.py install  # --single-version-externally-managed --record=record.txt
#$PYTHON setup.py build_ext -i
yes | pip install opencv-python
conda install -c conda-forge nb_conda_kernels -y
#conda upgrade notebook -y
pip install -e .

#mkdir nb_interface
cp -a $SRC_DIR/nb_interface/. $PREFIX/bin/nb_interface
echo "---------------- Finished build.sh ----------------"
