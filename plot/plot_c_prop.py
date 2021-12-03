import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import matplotlib.pylab  as pylab

# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize'  : 'xx-large',
         'axes.titlesize'  : 'xx-large',
         'xtick.labelsize' : 'xx-large',
         'ytick.labelsize' : 'xx-large'}

pylab.rcParams.update(params)

# use r raw strings when have slashes
path = r"C:\Users\cip19aac\Google Drive\Alovya_2021\code\HFV1_GPU_2D\HFV1_GPU_2D\results"

os.chdir(path)

sim_time_file = "clock_time_vs_sim_time.csv"
qx0_file      = "qx0-c-prop.csv"
qx1x_file     = "qx1x-c-prop.csv"
qx1y_file     = "qx1y-c-prop.csv"
qy0_file      = "qy0-c-prop.csv"
qy1x_file     = "qy1x-c-prop.csv"
qy1y_file     = "qy1y-c-prop.csv"

sim_time = pd.read_csv( os.path.join(path, sim_time_file) )
qx0      = pd.read_csv( os.path.join(path, qx0_file),  header=None )
qx1x     = pd.read_csv( os.path.join(path, qx1x_file), header=None )
qx1y     = pd.read_csv( os.path.join(path, qx1y_file), header=None )
qy0      = pd.read_csv( os.path.join(path, qy0_file),  header=None )
qy1x     = pd.read_csv( os.path.join(path, qy1x_file), header=None )
qy1y     = pd.read_csv( os.path.join(path, qy1y_file), header=None )

qx0_max  = qx0.abs().max(axis=1)
qx1x_max = qx1x.abs().max(axis=1)
qx1y_max = qx1y.abs().max(axis=1)
qy0_max  = qy0.abs().max(axis=1)
qy1x_max = qy1x.abs().max(axis=1)
qy1y_max = qy1y.abs().max(axis=1)

plt.figure()
plt.scatter(sim_time["sim_time"], qx0_max,  label='$q^0_x$',    marker='x')
plt.scatter(sim_time["sim_time"], qx1x_max, label='$q^{1x}_x$', marker='x')
plt.scatter(sim_time["sim_time"], qx1y_max, label='$q^{1y}_x$', marker='x')
plt.scatter(sim_time["sim_time"], qy0_max,  label='$q^0_y$',    marker='x')
plt.scatter(sim_time["sim_time"], qy1x_max, label='$q^{1x}_y$', marker='x')
plt.scatter(sim_time["sim_time"], qy1y_max, label='$q^{1y}_y$', marker='x')
plt.ticklabel_format(axis='x', style="sci")
plt.xlim(0, 100)
#plt.ylim(0, 1e-12)
#plt.set_yscale('log')
plt.legend()
plt.ylabel("Maximum error")
plt.xlabel("Simulation time (s)")
plt.savefig("c-prop", bbox_inches="tight")
#plt.show()
plt.clf()