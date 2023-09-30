This is a gaussian emulator and Baysiean analysis with PyMC <br />
 <br />

 Installation with anaconda (Should work on both linux and windows machine)
(Update 09/30/2023: Use mamba instead! Conda takes forever to solve environment!)
 ---
 
 1. Install anaconda/miniconda for python3 by following instruction on https://docs.conda.io/en/latest/miniconda.html.
 2. Update conda and add installation channel:
 ```
     $> conda update --all
     $> conda config --append channels conda-forge
 ```
 3. Install mamba and activate environment.
```
     $> conda install -c conda-forge mamba
     $> conda activate
```
 4. Go to /path/to/Bayesian-package and create virtual environment by running the following command:
 ```
     $> mamba env create --name <env_name> --file=environmentCrossPlatform.yml
 ```
 5. Activate the environment:
 ```
     $> conda activate <env_name>
 ```
 6. Start the program with 
 ```
    $> python -m GUI.GUIController.GUIControllerMP <number-of-cores>
 ```

IF the script refuses to use more than one core despite <number-of-cores> > 1, try to restart the terminal after step 4. Sometimes the installation of mamba messes up the CPU affinity. Restarting after installation usually fixes it.

If not, use 
```
    $> OPENBLAS_MAIN_FREE=1 python -m GUI.GUIController.GUIControllerMP <number-of-cores>
```

 Running GUI on HPCC
 ---
 
 This section only talks about running GUI in HPCC. Some changes to the installation procedure are needed because some version of MPI are not compatible with HPCC. 
 
TBD
 
 Using GUI
 ---

TBD
