**Installation**  
===================================================

Installation on MAC OS
----------------------

1. Download and install Anaconda <http://docs.continuum.io/anaconda/install> 

2. Clone these two repositories *in the same folder*:

    a. `git clone https://github.com/agiovann/Constrained_NMF.git`

    b. `git clone https://github.com/epnev/SPGL1_python_port.git`

3.     
    1. Go into the cloned folder and type `conda create --name CNMF --file requirements.txt`
    
    2. type `source activate CNMF` (this activates the environment, remember to do this every time you want to use the software)

    3. type `pip install tifffile picos` 

    4. type `conda install joblib` 

B. Use this in case 3. does not work. Type the following

        a. `git clone https://github.com/agiovann/Constrained_NMF.git`

        b. `git clone https://github.com/epnev/SPGL1_python_port.git`

        c. `conda create -n CNMF --no-deps ipython`

        d. `source activate CNMF`

        e. `conda install spyder numpy scipy ipyparallel matplotlib bokeh jupyter scikit-image scikit-learn joblib cvxopt`      

        f. `pip install picos`
        
        g. `pip install tifffile`

Test the system
----------------------

A. Using the Spyder IDE. 
    
    1. Open the file demo.py 

    2. Run the cells one by one inspecting the output

B. Using notebook. 

    1. type `ipython notebook`

    2. open the notebook called demoCNMF.ipynb and run cell by cell inspecting the result