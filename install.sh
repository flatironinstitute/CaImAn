#!/bin/bash
#to lauch to install the file
git clone https://github.com/simonsfoundation/CaImAn
cd CaImAn/
git pull
conda create -n CaImAn ipython --file requirements_conda.txt    
source activate CaImAn
pip install -r requirements_pip.txt
conda install -c menpo opencv3=3.1.0
python setup.py build_ext -i