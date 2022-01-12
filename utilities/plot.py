import os
import sys

def set_path(
    mode,
    testdir
):
    if (mode == "debug"):
        path = os.path.join(os.path.dirname(__file__), "..", "out", "build", "x64-Debug", testdir, "results")
    elif (mode == "release"):
        path = os.path.join(os.path.dirname(__file__), "..", "out", "build", "x64-Release", testdir, "results")
        
    return path

def EXIT_HELP():
    help_message = (
        "This tool is used to plot an x-against-y graph. Run using:\n" +
        "python plot.py <MODE> <TEST_CASE_FOLDER> <FILENAME> <X_AXIS_VAR> <Y_AXIS_VAR>, MODE=[debug,release]"
    )
    
    sys.exit(help_message)

if len(sys.argv) < 6:
    EXIT_HELP()

mode = sys.argv[1]

if mode == "debug" or mode == "release":
    import numpy             as np
    import pandas            as pd
    import matplotlib.pylab  as pylab
    import matplotlib.pyplot as plt
    
    dummy, mode, testdir, filename, xlabel, ylabel = sys.argv
    
    path = set_path(mode, testdir)
    
    print("Searching for data in path", path)
    
    data = pd.read_csv( os.path.join(path, filename) ).values.tolist()
    
    # from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
    params = {
        "legend.fontsize" : "xx-large",
        "axes.labelsize"  : "xx-large",
        "axes.titlesize"  : "xx-large",
        "xtick.labelsize" : "xx-large",
        "ytick.labelsize" : "xx-large"
    }
    
    pylab.rcParams.update(params)
    
    xlim = ( 1, len(data) )
    ylim = ( 0.95 * min(data), 1.05 * max(data) )
    
    plt.plot(data)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
else:
    EXIT_HELP()