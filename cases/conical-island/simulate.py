import os
import sys
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class ExperimentalDataConicalIsland:
    def __init__(self):
        print("Reading experimental data...")
        
        self.stages = [ "#" + str(i) for i in [6, 9, 16, 22] ]
        self.fields = ["time", "gauge_data"]
        self.data = {}
        
        for field in self.fields:
            self.data[field] = {}
            
            for stage in self.stages:
                self.data[field][stage] = {}
        
        for col, stage in enumerate(self.stages):
            cols = [2 * col, 2 * col + 1]
            experimental_dataframe = pd.read_excel("experimental.xls", header=None, skiprows=2, usecols=cols, engine="openpyxl")
            
            self.data["time"]      [stage] = experimental_dataframe.iloc[:,0]
            self.data["gauge_data"][stage] = experimental_dataframe.iloc[:,1]

class SimulationConicalIsland:
    def __init__(
            self,
            epsilons,
            solvers
        ):
            print("Creating fields for simulation results...")
            
            self.epsilons = epsilons
            self.solvers  = solvers
            self.fields   = ["simtime", "runtime", "gauge_data"]
            
            # stages 6, 9, 12 and 22 from
            # "Laboratory experiments of tsunami runup on a circular island"
            self.stages       = [ "#" + str(i) for i in [6, 9, 16, 22] ] 
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "cumulative-data.csv")
            self.results      = {}
            
            for solver in self.solvers:
                self.results[solver] = {}
                
                for epsilon in self.epsilons:
                    self.results[solver][epsilon] = {}
                
                    for field in self.fields:
                        self.results[solver][epsilon][field] = {}
                        
                    for stage in self.stages:
                        self.results[solver][epsilon]["gauge_data"][stage] = {}
                        
                for epsilon in epsilons:
                    #self.run(epsilon, solver)
                    
                    time_dataframe = pd.read_csv(self.runtime_file)
                    
                    self.results[solver][epsilon]["simtime"] = time_dataframe["simtime"]
                    self.results[solver][epsilon]["runtime"] = time_dataframe["runtime"]
                    
                    stage_dataframe = pd.read_csv(self.stage_file, skiprows=10, delimiter=" ", header=None)
                    
                    for i, stage in enumerate(self.stages):
                        self.results[solver][epsilon]["gauge_data"][stage] = stage_dataframe.iloc[:,i+1]
                    
    def run(
            self,
            epsilon,
            solver
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            input_file = "conical-island.par"
            
            with open(input_file, 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 10\n" +
                    "min_dt      1\n" +
                    "respath     results\n" +
                    "epsilon     %s\n" +
                    "fpfric      0.0\n" +
                    "rasterroot  conical-island\n" +
                    "stagefile   conical-island.stage\n" +
                    "tol_h       1e-3\n" +
                    "tol_q       0\n" +
                    "tol_s       1e-9\n" +
                    "g           9.80665\n" +
                    "massint     0.1\n" +
                    "sim_time    20\n" +
                    "solver      %s\n" +
                    "limitslopes off\n" +
                    "tol_Krivo   10\n" +
                    "cumulative  on\n" +
                    "wall_height 1"
                ) % (epsilon, solver)
                
                fp.write(params)
            
            executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
            
            subprocess.run( [os.path.join("..", executable), input_file] )
        
    def plot_exp_data(
        self,
        my_rc_params,
        exp_data,
    ):
        T = 6
        
        plt.rcParams.update(my_rc_params)
        
        fig = plt.figure( figsize=(6, 5) )
        
        gridspec = fig.add_gridspec(
            nrows=2,
            ncols=2,
            hspace=0.15,
            wspace=0.5
        )
        
        axs = gridspec.subplots()
        
        for ax, stage in zip(axs.flatten(), self.stages):
            for solver in self.solvers:
                for epsilon in self.epsilons:
                    if epsilon == 0:
                        label = "GPU-DG2" if solver == "mw" else "GPU-FV1"
                    elif np.isclose(epsilon, 1e-3):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                    elif np.isclose(epsilon, 1e-4):
                        label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                    
                    ax.plot(
                        self.results[solver][epsilon]["simtime"] + T,
                        self.results[solver][epsilon]["gauge_data"][stage] - self.results[solver][epsilon]["gauge_data"][stage][0],
                        linewidth=2.5,
                        label=label 
                    )
                
            ax.scatter(
                exp_data.data["time"]      [stage],
                exp_data.data["gauge_data"][stage],
                facecolor="None",
                edgecolor="black",
                label="Experimental"
            )
            
            if stage == "#6":
                ax.legend()
            
            ax.set_xlabel(r"$t \, (s)$")
            ax.set_ylabel(r"h + z \, $(m)$")
            ax.set_xlim(6, 20)
            axs[0,0].legend(
                bbox_to_anchor=(1.9, 1.4),
                ncol=2
            )
            
            xticks = [6, 10, 15, 20]
            
            ax.set_xticks( [] )
            ax.set_xticks(
                ticks=xticks,
                minor=False
            )
            
            ax.set_xticklabels(
                labels=xticks,
                minor=False
            )
            
        fig.savefig(os.path.join("results", "stages"), bbox_inches="tight")
        plt.close()
        
    def plot_speedups(
        self,
        my_rc_params
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots( figsize=(2.75, 2.5) )
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                runtime_ratio = self.results[solver][0]["runtime"] / self.results[solver][epsilon]["runtime"]
                
                if epsilon == 0:
                    label = "break-even"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"
                
                ax.plot(
                    self.results[solver][epsilon]["simtime"],
                    runtime_ratio,
                    linewidth=1    if epsilon == 0 else 2,
                    linestyle="-." if epsilon == 0 else "-",
                    color='k'      if epsilon == 0 else None,
                    label=label
                )
            
            xlim = (
                ( self.results[solver][0]["simtime"] ).iloc[0],
                ( self.results[solver][0]["simtime"] ).iloc[-1]
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
            "legend.fontsize" : "small"
        }
        
        self.plot_exp_data(my_rc_params, exp_data)
        self.plot_speedups(my_rc_params)
        
if __name__ == "__main__":
    subprocess.run( ["python", "stage.py" ] )
    subprocess.run( ["python", "raster.py"] )
    
    SimulationConicalIsland( [1e-3, 1e-4, 0], ["mw"] ).plot( ExperimentalDataConicalIsland() )