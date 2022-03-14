import os
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
            experimental_dataframe = pd.read_excel("experimental.xls", header=None, skiprows=2, usecols=cols)
            
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
            self.runtime_file = os.path.join("results", "simtime-vs-runtime.csv")
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
                    self.run(epsilon, solver)
                    
                    time_dataframe = pd.read_csv(self.runtime_file)
                    
                    self.results[solver][epsilon]["simtime"] = time_dataframe["simtime"]
                    self.results[solver][epsilon]["runtime"] = time_dataframe["runtime"]
                    
                    stage_dataframe = pd.read_csv(self.stage_file)
                    
                    for i, stage in enumerate(self.stages):
                        self.results[solver][epsilon]["gauge_data"][stage] = stage_dataframe.iloc[:,i+1]
                    
    def run(
            self,
            epsilon,
            solver
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            with open("conical-island.par", 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 10\n" +
                    "min_dt      1\n" +
                    "respath     .\\results\n" +
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
                    "limitslopes on\n" +
                    "cumulative  on\n" +
                    "wall_height 1"
                ) % (epsilon, solver)
                
                fp.write(params)
            
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "conical-island.par"] )
            
    def plot(
            self,
            exp_data
        ):
            T = 6
            
            my_rc_params = {
                "legend.fontsize" : "xx-large",
                "axes.labelsize"  : "xx-large",
                "axes.titlesize"  : "xx-large",
                "xtick.labelsize" : "xx-large",
                "ytick.labelsize" : "xx-large"
            }
            
            plt.rcParams.update(my_rc_params)
            
            print("Plotting stage data...")
            
            fig, ax = plt.subplots()
            
            for stage in self.stages:
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
                
                ax.set_xlabel(r"$t \, (s)$")
                ax.set_ylabel(r"Free surface elevation $(m)$")
                ax.set_xlim(6, 20)
                ax.legend()
                fig.savefig(os.path.join("results", "stage-" + stage), bbox_inches="tight")
                ax.clear()
            
            print("Plotting speedups...")
            
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
                        linewidth=2.5,
                        linestyle="--" if epsilon == 0 else "-",
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
        
if __name__ == "__main__":
    print("Writing stage file...")
    
    with open("conical-island.stage", 'w') as fp:
        stages = (
            "4\n" +
            "9.36  13.8\n" +
            "10.36 13.8\n" +
            "12.96 11.22\n" +
            "15.56 13.8"
        )
        
        fp.write(stages)
    
    subprocess.run( ["python", "raster.py"] )
    
    SimulationConicalIsland( [0, 1e-4, 1e-3], ["mw"] ).plot( ExperimentalDataConicalIsland() )