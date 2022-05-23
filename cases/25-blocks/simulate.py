import os
import sys
import subprocess
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

class ExperimentalData25Blocks:
    def __init__(self):
        print("Reading experimental data...")
        
        self.data = {}
        
        intervals = [4, 5, 6, 10]
        flow_vars = ["depth", "velocity"]
        
        for interval in intervals:
            if interval not in self.data:
                self.data[interval] = {}
                for var in flow_vars:
                    if var not in self.data[interval]:
                        self.data[interval][var] = {}
                        
        for interval in intervals:
            depth_file    = "WLxy_C1_" + str(interval) + "s.txt"
            velocity_file = "Vxy_C1_"  + str(interval) + "s.txt"
            
            self.data[interval]["depth"]    = pd.read_csv(depth_file,    header=None, delimiter=' ')
            self.data[interval]["velocity"] = pd.read_csv(velocity_file, header=None, delimiter=' ')
            
class Simulation25Blocks:
    def __init__(
            self,
            epsilons,
            solvers
        ):
            print("Reading raster header...")
            
            header = []
            
            with open("25-blocks.start", 'r') as fp:
                for i, line in enumerate(fp):
                    if i > 4: break
                    header.append(line)
            
            cast = lambda ans, i : int(ans) if i < 2 else float(ans)
            
            ncols, nrows, xmin, ymin, cellsize = [ cast(line.split()[1], i) for i, line in enumerate(header) ]
            
            print("Creating fields for simulation results...")
            
            x_range_start = 4 - xmin
            x_range_end   = 8 - xmin
            
            col_start = int(x_range_start / cellsize)
            col_end   = int(x_range_end   / cellsize)
            
            self.xmin         = xmin
            self.cellsize     = cellsize
            self.runtime_file = os.path.join("results", "cumulative-data.csv")
            self.epsilons     = epsilons
            self.solvers      = solvers
            self.flow_vars    = ("depth", "velocity")
            self.time_vars    = ("simtime", "runtime")
            self.i_range      = [ _ for _ in range(col_start,col_end) ]
            self.intervals    = [4, 5, 6, 10]
            self.results      = {}
            
            for solver in self.solvers:
                if solver not in self.results:
                    self.results[solver] = {}
                    for epsilon in self.epsilons:        
                        if epsilon not in self.results[solver]:
                            self.results[solver][epsilon] = {}
                            for var in self.time_vars:
                                if var not in self.results[solver][epsilon]:
                                    self.results[solver][epsilon][var] = {}
                            for interval in self.intervals:
                                if interval not in self.results[solver][epsilon]:
                                    self.results[solver][epsilon][interval] = {}
                                    for var in self.flow_vars:
                                        if var not in self.results[solver][epsilon][interval]:
                                            self.results[solver][epsilon][interval][var] = {}
            
            y_range = 0.2 - ymin
            
            row = int(y_range / cellsize)
            
            # reading from the top down
            # +6 to skip the 6 header lines
            self.slice_row = nrows - row + 6
            
            for solver in self.solvers:
                for epsilon in epsilons:
                    self.run(epsilon, solver)
                    
                    time_dataframe = pd.read_csv(self.runtime_file)
                    
                    self.results[solver][epsilon]["simtime"] = time_dataframe["simtime"]
                    self.results[solver][epsilon]["runtime"] = time_dataframe["runtime"]
                    
                    for interval in self.intervals:
                        h_raster_file  = os.path.join("results", "results-" + str(interval) + ".wd")
                        qx_raster_file = os.path.join("results", "results-" + str(interval) + ".qx")
                        qy_raster_file = os.path.join("results", "results-" + str(interval) + ".qy")
                        
                        self.results[solver][epsilon][interval]["depth"] = np.loadtxt(
                            fname=h_raster_file,
                            skiprows=self.slice_row,
                            max_rows=1,
                            usecols=self.i_range
                        )
                        
                        # calculating velocity from depth and discharge
                        vx = np.loadtxt(
                            fname=qx_raster_file,
                            skiprows=self.slice_row,
                            max_rows=1,
                            usecols=self.i_range
                        ) / self.results[solver][epsilon][interval]["depth"]
                        
                        vy = np.loadtxt(
                            fname=qy_raster_file,
                            skiprows=self.slice_row,
                            max_rows=1,
                            usecols=self.i_range
                        ) / self.results[solver][epsilon][interval]["depth"]
                        
                        self.results[solver][epsilon][interval]["velocity"] = np.sqrt(vx * vx + vy * vy)
                    
    def run(
            self,
            epsilon,
            solver
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            input_file = "25-blocks.par"
            
            with open(input_file, 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 11\n" +
                    "min_dt      1\n" +
                    "respath     results\n" +
                    "epsilon     %s\n" +
                    "fpfric      0.01\n" +
                    "rasterroot  25-blocks\n" +
                    "tol_h       1e-3\n" +
                    "tol_q       0\n" +
                    "tol_s       1e-9\n" +
                    "g           9.80665\n" +
                    "massint     0.01\n" +
                    "saveint     1\n" +
                    "sim_time    10\n" +
                    "limitslopes on\n" +
                    "tol_Krivo   1\n" +
                    "cumulative  on\n" +
                    "raster_out  on\n" +
                    "solver      %s\n" +
                    "wall_height 2.5"
                ) % (epsilon, solver)
                
                fp.write(params)
                
            executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
            
            subprocess.run( [os.path.join("..", executable), input_file] )
            
    def plot_exp_data(
        self,
        my_rc_params,
        exp_data
    ):
        print("Plotting depths and velocities...")
        
        plt.rcParams.update(my_rc_params)
        
        x = [self.xmin + i * self.cellsize for i in self.i_range]
        
        fig, ax = plt.subplots()
        
        for interval in self.intervals:
            for solver in self.solvers:
                for epsilon in self.epsilons:
                    if epsilon == 0:
                        label = "GPU-DG2" if solver == "mw" else "GPU-FV1"
                    elif np.isclose(epsilon, 1e-3):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                    elif np.isclose(epsilon, 1e-4):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                    
                    ax.plot(
                        x,
                        self.results[solver][epsilon][interval]["depth"],
                        label=label
                    )
            
            ax.scatter(
                exp_data.data[interval]["depth"].iloc[:][0],
                exp_data.data[interval]["depth"].iloc[:][1],
                label="Experimental",
                color="black"
            )
            
            ax.set_xlabel("$x \, (m)$")
            ax.set_ylabel("$Depth \, (m)$")
            ax.set_xlim(4, 8)
            ax.legend()
            fig.savefig(os.path.join("results", "depth-" + str(interval) ), bbox_inches="tight" )
            ax.clear()
            
            for solver in self.solvers:
                for epsilon in self.epsilons:
                    if epsilon == 0:
                        label = "GPU-DG2" if solver == "mw" else "GPU-FV1"
                    elif np.isclose(epsilon, 1e-3):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                    elif np.isclose(epsilon, 1e-4):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                    
                    ax.plot(
                        x,
                        self.results[solver][epsilon][interval]["velocity"],
                        label=label
                    )
            
            ax.scatter(
                exp_data.data[interval]["velocity"].iloc[:][0],
                exp_data.data[interval]["velocity"].iloc[:][1],
                label="Experimental",
                color="black"
            )
            
            ax.set_xlabel("$x \, (m)$")
            ax.set_ylabel("$Velocity \, (ms^{-1})$")
            ax.set_xlim(4, 8)
            ax.legend()
            fig.savefig(os.path.join("results", "velocity-" + str(interval) ), bbox_inches="tight")
            ax.clear()
            
        plt.close()
        
    def plot_speedups(
        self,
        my_rc_params
    ):
        print("Plotting speedups...")
        
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                runtime_ratio = self.results[solver][ self.epsilons[0] ]["runtime"] / self.results[solver][epsilon]["runtime"]
                
                if epsilon == 0:
                    label = "GPU-DG2" if solver == "mw" else "GPU-FV1"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                
                ax.plot(
                    self.results[solver][epsilon]["simtime"],
                    runtime_ratio,
                    linewidth=1    if epsilon == 0 else 2
                    linestyle="-." if epsilon == 0 else "-",
                    color='k'      if epsilon == 0 else None
                    label=label
                )
            
            xlim = (
                np.min( self.results[solver][ self.epsilons[0] ]["simtime"] ),
                np.max( self.results[solver][ self.epsilons[0] ]["simtime"] )
            )
            
            ax.set_xlim(xlim)
            ax.set_xlabel(r"$t \, (s)$")
            ax.set_ylabel( "Speedup ratio " + ("GPU-MWDG2/GPU-DG2" if solver == "mw" else "GPU-HWFV1/GPU-FV1") )
            ax.legend()
            fig.savefig(os.path.join("results", "runtimes-" + solver), bbox_inches="tight")
            ax.clear()
        
        plt.close()
        
    def plot(
        self,
        exp_data
    ):
        my_rc_params = {
            "legend.fontsize" : "large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large"
        }
        
        self.plot_exp_data(my_rc_params, exp_data)
        self.plot_speedups(my_rc_params)
        
    
if __name__ == "__main__":
    subprocess.run( ["python", "raster.py"] )
    
    Simulation25Blocks( [0, 1e-4, 1e-3], ["mw"] ).plot( ExperimentalData25Blocks() )