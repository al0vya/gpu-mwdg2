import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class DischargeErrors:
    def __init__(self, solver, relativepath=""):
        self.solver = solver;

        self.relativepath = relativepath

        if (relativepath == "debug"):
            self.relativepath = os.path.join("..", "out", "build", "x64-Debug", "test", "results")
        elif (relativepath == "release"):
            self.relativepath = os.path.join("..", "out", "build", "x64-Release", "test", "results")
    
        self.savepath = os.path.join(os.path.dirname(__file__), self.relativepath)

        sim_time_file = "clock_time_vs_sim_time.csv"
        qx0_file      = "qx0-c-prop.csv"
        qx1x_file     = "qx1x-c-prop.csv"
        qx1y_file     = "qx1y-c-prop.csv"
        qy0_file      = "qy0-c-prop.csv"
        qy1x_file     = "qy1x-c-prop.csv"
        qy1y_file     = "qy1y-c-prop.csv"
        
        self.sim_time = pd.read_csv( os.path.join(self.savepath, sim_time_file) )
        qx0           = pd.read_csv( os.path.join(self.savepath, qx0_file),  header=None )
        qx1x          = pd.read_csv( os.path.join(self.savepath, qx1x_file), header=None ) if solver == "mw" else None
        qx1y          = pd.read_csv( os.path.join(self.savepath, qx1y_file), header=None ) if solver == "mw" else None
        qy0           = pd.read_csv( os.path.join(self.savepath, qy0_file),  header=None )
        qy1x          = pd.read_csv( os.path.join(self.savepath, qy1x_file), header=None ) if solver == "mw" else None
        qy1y          = pd.read_csv( os.path.join(self.savepath, qy1y_file), header=None ) if solver == "mw" else None
        
        self.qx0_max  = qx0.abs().max(axis=1)
        self.qx1x_max = qx1x.abs().max(axis=1) if solver == "mw" else None
        self.qx1y_max = qx1y.abs().max(axis=1) if solver == "mw" else None
        self.qy0_max  = qy0.abs().max(axis=1)
        self.qy1x_max = qy1x.abs().max(axis=1) if solver == "mw" else None
        self.qy1y_max = qy1y.abs().max(axis=1) if solver == "mw" else None

    def plot_errors(self, test_number, test_name):

        print("Plotting maximum discharge errors for test %s..." % test_name)

        plt.figure()
        
        plt.scatter(self.sim_time["sim_time"], self.qx0_max,  label='$q^0_x$', marker='x')
        plt.scatter(self.sim_time["sim_time"], self.qy0_max,  label='$q^0_y$', marker='x')
        
        if self.solver == "mw":
            plt.scatter(self.sim_time["sim_time"], self.qx1x_max, label='$q^{1x}_x$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qx1y_max, label='$q^{1y}_x$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qy1x_max, label='$q^{1x}_y$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qy1y_max, label='$q^{1y}_y$', marker='x')
        
        xlim = self.sim_time["sim_time"][0], self.sim_time["sim_time"][-1]

        plt.ticklabel_format(axis='x', style="sci")
        plt.xlim(xlim)
        plt.legend()
        plt.ylabel("Maximum error")
        plt.xlabel("Simulation time (s)")

        filename = str(test_number) + "-c-prop-" + test_name

        plt.savefig(os.path.join(self.savepath, filename), bbox_inches="tight")
        
        plt.clf()