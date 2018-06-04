import cPickle as pickle
import pandas as pd
from numpy import pi, sin
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

with open('result/e120_LOO.pkl', 'rb') as buff:
    data = pickle.load(buff)

emulator, trace  = data['model'], data['trace']
dataloader = data['data']['data']
prior = data['prior']
#dataloader.ChangeExp('data/e120/e120_validation4.csv')
exp_data, exp_err = dataloader.exp_result, np.diag(np.sqrt(dataloader.exp_cov))


prior_range = {}
ini_par = []
for par_name in list(prior):
    prior_range[par_name] = (prior[par_name][0], prior[par_name][1])
    ini_par.append(0.5*(prior[par_name][0] + prior[par_name][1]))

ini_par = np.array(ini_par)
# transform input by input_pipe and put it in our emulator
result, var = emulator.Emulate(ini_par)
# need to transform back to our output space by output_pipe
num_output = result.shape[0]

with open('result/e120_bugfix_model_new.pkl', 'rb') as buff:
    data = pickle.load(buff)

emulator_RBF  = data['model']

def signal(par):
    result, var = emulator.Emulate(par)
    return result, var

def signal_RBF(par):
    result, var = emulator_RBF.Emulate(par)
    return result, var

axis_color = 'lightgoldenrodyellow'

fig = plt.figure()
ax = fig.add_subplot(111)

# Adjust the subplots region to leave some space for the sliders and buttons
fig.subplots_adjust(left=0.25, bottom=0.25)

t = np.arange(0, num_output)
new_par = ini_par

# Draw the initial plot
# The 'line' variable is used for modifying the line later
ax.plot(t, exp_data, 'ro', linewidth=2)
result, var = signal(ini_par)
err = ax.errorbar(t, result, yerr=np.sqrt(np.diag(var)), linewidth=2, color='red', label='Before Bug fix')
line, _, (bars,) = err

result, var = signal_RBF(ini_par)
line_RBF, _, (bars_RBF,) = ax.errorbar(t, result, yerr=np.sqrt(np.diag(var)), linewidth=2, color='blue', label='After Bug fix')

ax.set_xlim([-1, num_output+1])
ax.set_ylim([0.5, 1.9])
ax.legend(loc='upper left')

# Add two sliders for tweaking the parameters

# Define an axes area and draw a slider in it
amp_slider = []
for index, par_name in enumerate(list(prior)):
    amp_slider_ax  = fig.add_axes([0.25, 0.1 + 0.05*index, 0.65, 0.03])
    amp_slider.append(Slider(amp_slider_ax, par_name, prior_range[par_name][0], prior_range[par_name][1], valinit=new_par[index]))


# Define an action for modifying the line when any slider's value changes
def sliders_on_changed(val):
    par = []
    for slider in amp_slider:
        par.append(slider.val)
    ydata, var = signal(np.array(par))
    err = np.sqrt(np.diag(var))

    yerr_top = ydata + err
    yerr_bot = ydata - err
    x_base = line.get_xdata()
    new_segments = [np.array([[x, yt], [x, yb]]) for
                    x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
    line.set_ydata(ydata)
    bars.set_segments(new_segments)


    ydata, var = signal_RBF(np.array(par))
    #line_RBF.set_ydata(ydata)
   
    yerr_top = ydata + err
    yerr_bot = ydata - err
    new_segments = [np.array([[x, yt], [x, yb]]) for
                    x, yt, yb in zip(x_base, yerr_top, yerr_bot)]
    line_RBF.set_ydata(ydata)
    bars_RBF.set_segments(new_segments)


    fig.canvas.draw_idle()

for slider in amp_slider:
    slider.on_changed(sliders_on_changed)

"""
# Add a button for resetting the parameters
reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
reset_button = Button(reset_button_ax, 'Reset', color=axis_color, hovercolor='0.975')
def reset_button_on_clicked(mouse_event):
    freq_slider.reset()
    amp_slider.reset()
reset_button.on_clicked(reset_button_on_clicked)

# Add a set of radio buttons for changing color
color_radios_ax = fig.add_axes([0.025, 0.5, 0.15, 0.15])
color_radios = RadioButtons(color_radios_ax, ('red', 'blue', 'green'), active=0)
def color_radios_on_clicked(label):
    line.set_color(label)
    fig.canvas.draw_idle()
color_radios.on_clicked(color_radios_on_clicked)
"""
plt.show()
