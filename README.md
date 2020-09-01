This is a gaussian emulator and Baysiean analysis with PyMC <br />
 <br />

 If you are installing on Windows
 ---
 
 Do the following before proceeding to "Installation with anaconda". If you are using Linux, ignore this section and proceed directly to the next.

 1. Install Microsoft MPI by following instructions on https://docs.microsoft.com/en-us/message-passing-interface/microsoft-mpi

 2. Install MS visual studio, community version is enough, on https://visualstudio.microsoft.com/ 
 
 Installation with anaconda
 ---
 
 1. Install anaconda/miniconda for python3 by following instruction on https://docs.conda.io/en/latest/miniconda.html.
 2. Update conda and add installation channel:
 ```
     $> conda update --all
     $> conda config --append channels conda-forge
 ```
 2. Go to /path/to/Bayesian-package and create virtual environment by running the following command:
 ```
     $> conda env create -f environment.yml
 ```   
     Or, if you are using Windows,
 ```
     $> conda env create -f environmentCrossPlatform.yml
 ```
 3. Activate the environment:
 ```
     $> conda activate Bayesian
 ```
 4. Start the program with 
 ```
     $> ./Bayesian.sh <number of cores>
 ```
 Or, if you are using Windows,
 ```
    $> mpiexec -n <number of cores + 1> python -m GUI.GUIController.GUIController
 ```

 Running GUI on HPCC
 ---
 
 This section only talks about running GUI in HPCC. Some changes to the installation procedure are needed because some version of MPI are not compatible with HPCC. 
 
TBD
 
 Using GUI
 ---

TBD
