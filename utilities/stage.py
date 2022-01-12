import os
import numpy             as np
import pandas            as pd
import matplotlib.pylab  as pylab
import matplotlib.pyplot as plt

def plot_stage(
        i,
        ax,
        stages,
        xlim
    ):
        stage = "stage" + str(i + 1)
        
        ax.plot(stages["time"], stages[stage], linewidth=2)
        
        ylim = ( np.min( stages[stage] ), np.max( stages[stage] ) )
        
        plt.setp(ax, xlabel="$t \, (s)$", ylabel="$h \, (m)$", xlim=xlim, ylim=ylim)
        
        plt.show()
    
# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
params = {
    "legend.fontsize" : "xx-large",
    "axes.labelsize"  : "xx-large",
    "axes.titlesize"  : "xx-large",
    "xtick.labelsize" : "xx-large",
    "ytick.labelsize" : "xx-large"
}

pylab.rcParams.update(params)

path = os.path.dirname(__file__)

stagefile = "stage.wd"

stages = pd.read_csv( os.path.join(path, stagefile ) )

num_stages = stages.shape[1] - 1

xlim = (np.min( stages["time"] ), np.max( stages["time"] ))

nrows = int( np.ceil( np.sqrt(num_stages) ) )
ncols = nrows

fig, axs = plt.subplots(nrows=nrows, ncols=ncols)

fig.tight_layout()

if num_stages == 1:
    plot_stage(i=0, ax=axs, stages=stages, xlim=xlim)
else:
    for i, ax in enumerate(axs.flat):
        if i + 1 > num_stages: continue
        
        plot_stage(i=i, ax=ax, stages=stages, xlim=xlim)