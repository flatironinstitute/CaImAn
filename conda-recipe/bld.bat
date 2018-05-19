pip install opencv-python
conda install -c conda-forge nb_conda_kernels -y

pip install -e .

conda install numba

xcopy %SRC_DIR%\nb_interface\. %PREFIX%\bin\nb_interface
