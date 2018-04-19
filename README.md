This is a gaussian emulator and Baysiean analysis with PyMC/PyMC3 <br />
 <br />
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
&nbsp;&nbsp;&nbsp;&nbsp;3 files are needed for the script to run, which are: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Parameter prior <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model value <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Experimental value <br />
&nbsp;&nbsp;&nbsp;&nbsp;They must be in the form of csv. The first row must be the name of each variables. <br />
&nbsp;&nbsp;&nbsp;&nbsp;It is highly recommanded that all the files be put into directory 'data' <br />
&nbsp;&nbsp;&nbsp;&nbsp;For Parameter prior, the format will be: <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;par_name1, par_name2, par_name3, ... ,par_namen <br />
&nbsp;&nbsp;&nbsp;&nbsp;lowerlimit1, lowerlimit2, ....., lowerlimitn <br />
&nbsp;&nbsp;&nbsp;&nbsp;higherlimit1, higherlimit2, ....., higherlimitn <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;For Model value, the format will be: <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;par_name1, par_name2, ..., par_namen, var_name1, var_name2, .... , var_namem, var_name1_Error, var_name2_Error, ...., var_namem_Error <br />
&nbsp;&nbsp;&nbsp;&nbsp;corresponding values delimited by comma <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;variables with '_Error' at the end will be treated as the error bar size. This is optional, but if this is missing validation cannot be performed. <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;For experimental values, the format is identical to model value without all the par_name: <br />
&nbsp;&nbsp;&nbsp;&nbsp; <br />
&nbsp;&nbsp;&nbsp;&nbsp;var_name1, var_name2, .... , var_namem, var_name1_Error, var_name2_Error, ...., var_namem_Error <br />
&nbsp;&nbsp;&nbsp;&nbsp;corresponding values delimited by comma. For details, please read the examples included in data/TestExample/ <br />
 <br />
1.1 Convert MADAI data format to this emulator's format <br />
&nbsp;&nbsp;&nbsp;&nbsp; <br />
&nbsp;&nbsp;&nbsp;&nbsp;Most of our simulation results are stored in MADAI format, which is: <br />
&nbsp;&nbsp;&nbsp;&nbsp;model_output <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-run0000 <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-parameters.dat <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-results.dat <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-run0001 <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-parameters.dat <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-results.dat <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;. <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-run00xx <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-parameters.dat <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;|-results.dat <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;Under directory DataReader, there is a read_data.py script which convert all of the above file structure into a single csv file. <br />
&nbsp;&nbsp;&nbsp;&nbsp;To use the script, enter: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python read_data.py /path/to/model_output sub_your_name <br />
&nbsp;&nbsp;&nbsp;&nbsp;The output will be sub_your_name.csv. You should move it manually to the designated folder of your projects  <br />
&nbsp;&nbsp;&nbsp;&nbsp;Experimental and parameter prior conversions are still absent. Hopefully it will be added back later <br />
 <br />
2. Optimize the Emulator <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;With all our data, it is necessary to Optimize the value of hyperparameters before using the emulator. It is done through either maximizing pesudo log-likelihood with LOO-CV or marginal log-likelihood. To choose one, change the training behaviour in member Train in class EmulatorMultiOutput under Emulator/Emulator.py. Training speed can be changed there as well. <br />
&nbsp;&nbsp;&nbsp;&nbsp;Training is done through the following command: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python Training data/directory_to_model/prior.csv data/directory_to_model/model_data.csv data/directory_to_model/exp_data.csv sub_your_name <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;It will generate a pickle files containing the optimized emulator (loaded with model data) and all the priors and exp_result in training/sub_your_name.pkl (By default, all trainning files will be stored under directory 'training'. Bayesian analysis will be performed with emulator inside the pickle file.  <br />
&nbsp;&nbsp;&nbsp;&nbsp;Here you will also define the number of PCA components to be used. The integer given to the constructor of class PCA is the number of components used. Change that number if you want and it will be saved in the pickle file. <br />
 <br />
3. (Optional) Validation <br />
&nbsp;&nbsp;&nbsp;&nbsp;You can see how good the emulator is by running the validation run. It will pull some training points out of the emulator and pretend that the left out point is the "experimental value". We will see how good it recreate the correct value with Bayesian analysis. To do this, use the following command: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python StatLOO.py training/sub_your_name.pkl '5' output_name <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;'5' represent which run do you want to pull out. It can be anything from 0 to num_training_points - 1. It will generate another pickle file and a csv file: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;result/output_name.pkl and output_name.csv <br />
&nbsp;&nbsp;&nbsp;&nbsp;The pickle files contains the pandas dataframe with trace from MCMC. output_name.csv contains the same information but in a csv file format. They will be needed for plotting posteriors. At the end of the script, the correlation plots along with the corresponding exact value (white inverted triangle) are shown. <br />
&nbsp;&nbsp;&nbsp;&nbsp;CAUCTION: Sometimes the initial value of MCMC may not be good and the burn in lenght is not long enough either. It will appears as if all the data are concentrated in 1 pixel. If that's the case, please re-run the script. The initial seed is chosen at random so running the same script again will generate different results and hopefull the result will be better <br />
 <br />
4. Generate Posterior <br />
&nbsp;&nbsp;&nbsp;&nbsp; <br />
&nbsp;&nbsp;&nbsp;&nbsp;Analysis is done with the following command: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python StatParallel.py training/sub_your_name.pkl output_name <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;The result is very similar to what is included in step 3, except that no exact value in parameter space is drawn and experimental values are used. <br />
&nbsp;&nbsp;&nbsp;&nbsp;It will also generate a csv and pkl file. For details please refer to step 3 <br />
 <br />
5. Understand the posterior <br />
&nbsp;&nbsp;&nbsp;&nbsp; <br />
&nbsp;&nbsp;&nbsp;&nbsp;We can redraw the posterior with PlotCorrelation.py: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python PlotCorrelation.py result/output_name.pkl <br />
 <br />
&nbsp;&nbsp;&nbsp;&nbsp;We can also draw the constraint on output variables instead of input variables by PlotPosterior.py: <br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;python PlotPosterior.py <br />
&nbsp;&nbsp;&nbsp;&nbsp;It will first show variables draw by sampling prior, then another plot with samples from posterior. The black contour represent the 95% confidence interval. <br />
&nbsp;&nbsp;&nbsp;&nbsp; <br />
