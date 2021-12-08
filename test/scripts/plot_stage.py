import pandas            as pd
import matplotlib.pyplot as plt
import numpy             as np
import sys
import os
import matplotlib.pylab  as pylab

# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
params = {
'legend.fontsize' : 'xx-large',
'axes.labelsize'  : 'xx-large',
'axes.titlesize'  : 'xx-large',
'xtick.labelsize' : 'xx-large',
'ytick.labelsize' : 'xx-large'
}

pylab.rcParams.update(params)

# use r raw strings when have slashes
path = r"C:\Users\alovy\Documents\HFV1_GPU_2D\HFV1_GPU_2D\results"

os.chdir(path)

stagefile = "stage.wd"

stages_adaptive_3 = pd.read_csv( os.path.join(path, stagefile ) )

plt.plot(stages_adaptive_3["time"], stages_adaptive_3["stage1"], linewidth=2)
#plt.xlim([0, 22.5])
#plt.ylim([-0.01, 0.045])
plt.xlabel("Time (s)")
plt.ylabel("h (m)")
plt.legend()

#plt.show()
plt.savefig("stage", bbox_inches="tight")