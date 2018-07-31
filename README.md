This is a gaussian emulator and Baysiean analysis with PyMC <br />
 <br />
 
 Installation with anaconda
 ---
 
 Please use anaconda python package. Installation with pip and virtualenv is possible but it will be hard to get certain packages (like pymc and wxpython) installed. <br />
 <br />
 For anaconda python:
 
 1. Install anaconda for python2 by following instructions on https://conda.io/docs/user-guide/tasks/manage-pkgs.html#installing-packages. 
 2. Go to /path/to/Bayesian-package and create virtual environment by running the following command:
 ```
     $> conda create --name myPy --file requirements.txt
 ```
 This script is tested for python 2.7.15 <br />
 3. Install autograd with pip. Unfortunately this package is not available on anaconda so you will have to install it with pip:
 ```
     $> pip install autograd
 ```     
 4. Run GUI.py if you want GUI, or any individual calculation scripts otherwise (explained below).
 ```
     $> python GUI.py &
 ```   
 and you are ready to go.
 
 Installation with pip
 ---
 
 Please be warned that pip installation may not be successful depending on your avaliable libraries. It is highly recommended that anaconda installation be used.
 
 1. Create a virtual environment with virtualenv and activate it inside /path/to/Bsyesian-package
```
    $> virtualenv myPy
    $> source myPy/bin/activate
```
 2. Install necessay packages with the following commands. If you can install pymc successfully, skip step 3.
 ```
     $> pip install numpy
     $> pip install scipy
     $> pip install matplotlib
     $> pip install pandas
     $> pip install pymc
 ```
 3. Pymc is difficult to install as it relies on external gfortran code. Fortran libraries may not be compiled correctly and there's nothing pip can do. Serval solutions are as follows:
     * Try to install liblapack-dev with 
     ```
        $> sudo apt-get liblapack-dev
     ```
     * Load liblapack-dev with module load
     * Alternatively, local installation of liblapack-dev may work but I have not luck with it
     * Switch to compatible gcc and gfortran compiler
     
 If pymc cannot be installed correctly, anaconda build may be the only solution.

 4. Install wxpython. Unfortunately wxpython is known to be defective in virtualenv so you have to install wxpython OUTSIDE virtualenv first and link the installed libraries against your virtual environment:
 ```
    $> ln -s /usr/lib/python2.7/dist-packages/wx.pth myPy/lib/python2.7/site-packages/
    $> ln -s /usr/lib/python2.7/dist-packages/wx-3.0-gtk2 myPy/lib/python2.7/site-packages/
```
5. Run GUI.py, or any individual calculation scripts otherwise (explained below).
 ```
     $> python GUI.py &
 ```   
 
 Running GUI on HPCC
 ---
 
 This section only talks about running GUI in HPCC. If traditional command is used, then just submit jobs like normal and wait for it to run (see details below).
 
 While running on fishtank maybe fast enough, it could be a bad idea to use up everyone's CPU power. Running on HPCC could be a good idea. Installation process is the same, but you need to line up in a queue for CPU power.
 
 1. Get into a cluster and Install like normal
 2. Ask for CPU time:
 ```
     $> qsub -I -X -l nodes=1:ppn=15,walltime=12:00:00,mem=4G -N Init
 ```
 ppn is the number of cores and mem is the memory. It is recommended that at least 2G of memory be used. Increasing number of nodes will not help due to shortcommings in the program design. 
 3. Wait for the process to start, cd into path/to/Bayesian-package, activate virtual environment of your choice and use GUI as normal.
 
 
 
 
 
 
In the following, I will use the following naming convention for each variables: <br />
 <br />
par : parameters, i.e. the input of the theoretical function <br />
var : variables, i.e. output of the theoretical function <br />



 <br />
To use the code, follow the following steps: <br />
 <br />
 <br />
1. Generate training points. <br />
 <br />
    3 files are needed for the script to run, which are: <br />
        Parameter prior <br />
        Model value <br />
        Experimental value <br />
    They must be in the form of csv. The first row must be the name of each variables. <br />
    It is highly recommanded that all the files be put into directory 'data' <br />
    For Parameter prior, the format will be: <br />
 <br />
    par_name1, par_name2, par_name3, ... ,par_namen <br />
    lowerlimit1, lowerlimit2, ....., lowerlimitn <br />
    higherlimit1, higherlimit2, ....., higherlimitn <br />
 <br />
    For Model value, the format will be: <br />
 <br />
    par_name1, par_name2, ..., par_namen, var_name1, var_name2, .... , var_namem, var_name1_Error, var_name2_Error, ...., var_namem_Error <br />
    corresponding values delimited by comma <br />
 <br />
    variables with '_Error' at the end will be treated as the error bar size. This is optional, but if this is missing validation cannot be performed. <br />
 <br />
    For experimental values, the format is identical to model value without all the par_name: <br />
     <br />
    var_name1, var_name2, .... , var_namem, var_name1_Error, var_name2_Error, ...., var_namem_Error <br />
    corresponding values delimited by comma. For details, please read the examples included in data/TestExample/ <br />
 <br />
1.1 Convert MADAI data format to this emulator's format <br />
     <br />
    Most of our simulation results are stored in MADAI format, which is: <br />
    model_output <br />
        | <br />
        |-run0000 <br />
        |    |-parameters.dat <br />
        |    |-results.dat <br />
        |-run0001 <br />
        |    |-parameters.dat <br />
        |    |-results.dat <br />
        . <br />
        . <br />
        . <br />
        |-run00xx <br />
        |    |-parameters.dat <br />
        |    |-results.dat <br />
 <br />
    Under directory DataReader, there is a read_data.py script which convert all of the above file structure into a single csv file. <br />
    To use the script, enter: <br />
        python read_data.py /path/to/model_output sub_your_name <br />
    The output will be sub_your_name.csv. You should move it manually to the designated folder of your projects  <br />
    Experimental and parameter prior conversions are still absent. Hopefully it will be added back later <br />
 <br />
2. Optimize the Emulator <br />
 <br />
    With all our data, it is necessary to Optimize the value of hyperparameters before using the emulator. It is done through either maximizing pesudo log-likelihood with LOO-CV or marginal log-likelihood. To choose one, change the training behaviour in member Train in class EmulatorMultiOutput under Emulator/Emulator.py. Training speed can be changed there as well. <br />
    Training is done through the following command: <br />
        python Training data/directory_to_model/prior.csv data/directory_to_model/model_data.csv data/directory_to_model/exp_data.csv sub_your_name <br />
 <br />
    It will generate a pickle files containing the optimized emulator (loaded with model data) and all the priors and exp_result in training/sub_your_name.pkl (By default, all trainning files will be stored under directory 'training'. Bayesian analysis will be performed with emulator inside the pickle file.  <br />
    Here you will also define the number of PCA components to be used. The integer given to the constructor of class PCA is the number of components used. Change that number if you want and it will be saved in the pickle file. <br />
 <br />
3. (Optional) Validation <br />
    You can see how good the emulator is by running the validation run. It will pull some training points out of the emulator and pretend that the left out point is the "experimental value". We will see how good it recreate the correct value with Bayesian analysis. To do this, use the following command: <br />
        python StatLOO.py training/sub_your_name.pkl '5' output_name <br />
 <br />
    '5' represent which run do you want to pull out. It can be anything from 0 to num_training_points - 1. It will generate another pickle file and a csv file: <br />
        result/output_name.pkl and output_name.csv <br />
    The pickle files contains the pandas dataframe with trace from MCMC. output_name.csv contains the same information but in a csv file format. They will be needed for plotting posteriors. At the end of the script, the correlation plots along with the corresponding exact value (white inverted triangle) are shown. <br />
    CAUCTION: Sometimes the initial value of MCMC may not be good and the burn in lenght is not long enough either. It will appears as if all the data are concentrated in 1 pixel. If that's the case, please re-run the script. The initial seed is chosen at random so running the same script again will generate different results and hopefull the result will be better <br />
 <br />
4. Generate Posterior <br />
     <br />
    Analysis is done with the following command: <br />
        python StatParallel.py training/sub_your_name.pkl output_name <br />
 <br />
    The result is very similar to what is included in step 3, except that no exact value in parameter space is drawn and experimental values are used. <br />
    It will also generate a csv and pkl file. For details please refer to step 3 <br />
 <br />
5. Understand the posterior <br />
     <br />
    We can redraw the posterior with PlotCorrelation.py: <br />
        python PlotCorrelation.py result/output_name.pkl <br />
 <br />
    We can also draw the constraint on output variables instead of input variables by PlotPosterior.py: <br />
        python PlotPosterior.py <br />
    It will first show variables draw by sampling prior, then another plot with samples from posterior. The black contour represent the 95% confidence interval. <br />
     <br />
<br />
Before installing requirements.txt in a virtual environment, you need to:<br />
ln -s /usr/lib/python2.7/dist-packages/wx.pth myPy/lib/python2.7/site-packages/<br />
ln -s /usr/lib/python2.7/dist-packages/wx-3.0-gtk2 myPy/lib/python2.7/site-packages/<br />
Because wx cannot be installed easily<br />
