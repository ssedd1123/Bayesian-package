This is a gaussian emulator and Baysiean analysis with PyMC <br />
 <br />

 Installation with miniforge
 ---

 1. Install miniforge for python3 by following the instructions on https://github.com/conda-forge/miniforge.
 2. Update and activate the base environment:
 ```
     $> conda update --all
     $> conda activate
 ```
 3. Go to /path/to/Bayesian-package and create the virtual environment.

    On Linux/Windows:
 ```
     $> conda env create --name <env_name> --file=environmentCrossPlatform.yml
 ```

    On Mac, `environmentCrossPlatform.yml` does NOT work — use `environmentMac.yml` instead, and prefix the command with `CONDA_SUBDIR=osx-64` so the correct platform packages are selected (without it, packages will fail to install):

 ```
     $> CONDA_SUBDIR=osx-64 conda env create --name <env_name> --file=environmentMac.yml
 ```
 4. Activate the environment:
 ```
     $> conda activate <env_name>
 ```
 5. Start the program.

    On Linux/Windows:
 ```
    $> python -m GUI.GUIController.GUIControllerMP <number-of-cores>
 ```

    On Mac, you must use `pythonw` instead of `python` (the regular `python` will not work for the GUI):

 ```
    $> pythonw -m GUI.GUIController.GUIControllerMP <number-of-cores>
 ```

IF the script refuses to use more than one core despite <number-of-cores> > 1, try to restart the terminal after step 3. Sometimes the installation of conda messes up the CPU affinity. Restarting after installation usually fixes it.

If not, use
```
    $> OPENBLAS_MAIN_FREE=1 python -m GUI.GUIController.GUIControllerMP <number-of-cores>
```
(or `pythonw` instead of `python` on Mac).

 Using GUI
 ---

TBD
