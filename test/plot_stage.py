import os
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
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

path = os.path.dirname(__file__)

stagefile = "stage.wd"

stages_adaptive_3 = pd.read_csv( os.path.join(path, stagefile ) )

plt.plot(stages_adaptive_3["time"], stages_adaptive_3["stage1"], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("h (m)")
plt.legend()

plt.show()