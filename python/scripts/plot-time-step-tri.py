# this script plots the time steps for the triangular dam break in the x and y directions

import os

import matplotlib.pyplot as plt
import pandas            as pd
import numpy             as np

# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
import matplotlib.pylab  as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize'  : 'xx-large',
         'axes.titlesize'  : 'xx-large',
         'xtick.labelsize' : 'xx-large',
         'ytick.labelsize' : 'xx-large'}

pylab.rcParams.update(params)

path = r"C:\Users\alovy\Documents\HFV1_GPU_2D\HFV1_GPU_2D\results\tests"
os.chdir(path)

file_name = "triangular-dam-break-x-vs-y.csv"

sim_time_y  = pd.read_csv( os.path.join(path, file_name) )["sim-time-y"]
sim_time_x  = pd.read_csv( os.path.join(path, file_name) )["sim-time-x"]
time_step_y = pd.read_csv( os.path.join(path, file_name) )["time-step-y"]
time_step_x = pd.read_csv( os.path.join(path, file_name) )["time-step-x"]

x_min = min( sim_time_x.min(),  sim_time_y.min() )
y_min = min( time_step_x.min(), time_step_y.min() )

x_max = max( sim_time_x.max(),  sim_time_y.max() )
y_max = max( time_step_x.max(), time_step_y.max() )

plt.scatter(sim_time_x, time_step_x, s=10, label="x direction")
plt.scatter(sim_time_y, time_step_y, s=10, label="y direction")
plt.ylabel("Time step (s)")
plt.xlabel("Simulation time (s)")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.savefig("x-vs-y-tri", bbox_inches="tight")

