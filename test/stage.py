import os
import numpy             as np
import pandas            as pd
import matplotlib.pylab  as pylab
import matplotlib.pyplot as plt

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

stage = pd.read_csv( os.path.join(path, stagefile ) )

plt.plot(stage["time"], stage["stage1"], linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("h (m)")
plt.legend()

plt.show()