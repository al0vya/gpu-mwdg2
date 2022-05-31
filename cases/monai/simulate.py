import os
import sys
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class ExperimentalDataMonai:
    def __init__(self):
        print("Reading experimental data...")
        
        experimental_dataframe = pd.read_csv("experimental.txt")
        
        self.time       = experimental_dataframe["time"]
        self.gauge_data = experimental_dataframe["stage1"]

class SimulationMonai:
    def __init__(
            self,
            epsilons,
            solvers
        ):
            print("Creating fields for simulation results...")
            
            self.solvers      = solvers
            self.epsilons     = epsilons
            self.fields       = ["simtime", "runtime", "gauge_data"]
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "cumulative-data.csv")
            self.results      = {}
            
            for solver in self.solvers:
                self.results[solver] = {}
                
                for epsilon in self.epsilons:
                    if epsilon not in self.results:
                        self.results[solver][epsilon] = {}
                        for field in self.fields:
                            if field not in self.results[solver][epsilon]:
                                self.results[solver][epsilon][field] = {}
                                
                for epsilon in epsilons:
                    self.run(epsilon, solver)
                    
                    time_dataframe = pd.read_csv(self.runtime_file)
                    
                    self.results[solver][epsilon]["simtime"]    = time_dataframe["simtime"]
                    self.results[solver][epsilon]["runtime"]    = time_dataframe["runtime"]
                    self.results[solver][epsilon]["gauge_data"] = pd.read_csv(
                        self.stage_file,
                        skiprows=7,
                        delimiter=" ",
                        header=None
                    ).iloc[:,1]
            
    def run(
            self,
            epsilon,
            solver
        ):
            print("Running simulation, eps = " + str(epsilon) + ", solver: " + solver)
            
            input_file = "monai.par"
            
            with open(input_file, 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 9\n" +
                    "min_dt      1\n" +
                    "respath     results\n" +
                    "epsilon     %s\n" +
                    "fpfric      0.01\n" +
                    "rasterroot  monai\n" +
                    "bcifile     monai.bci\n" +
                    "bdyfile     monai.bdy\n" +
                    "stagefile   monai.stage\n" +
                    "tol_h       1e-3\n" +
                    "tol_q       0\n" +
                    "tol_s       1e-9\n" +
                    "g           9.80665\n" +
                    "massint     0.1\n" +
                    "sim_time    22.5\n" +
                    "solver      %s\n" +
                    "limitslopes off\n" +
                    "tol_Krivo   10\n" +
                    "cumulative  on\n" +
                    "wall_height 0.5"
                ) % (epsilon, solver)
                
                fp.write(params)
            
            executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
            
            subprocess.run( [os.path.join("..", executable), input_file] )
            
    def plot_exp_data(
        self,
        my_rc_params,
        exp_data
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
        for solver in self.solvers:
            for epsilon in self.epsilons:
                if epsilon == 0:
                    label = "GPU-DG2" if solver == "mw" else "GPU-FV1"
                elif np.isclose(epsilon, 1e-3):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-3}$"
                elif np.isclose(epsilon, 1e-4):
                    label = ("GPU-MWDG2" if solver == "mw" else "GPU-HWFV1") + r", $\epsilon = 10^{-4}$"

                ax.plot(
                    self.results[solver][epsilon]["simtime"],
                    self.results[solver][epsilon]["gauge_data"],
                    linewidth=2.5,
                    label=label
                )
            
        ax.scatter(
            exp_data.time,
            exp_data.gauge_data,
            facecolor="None",
            edgecolor="black",
            label="Experimental"
        )
        
        ax.set_xlabel(r"$t \, (s)$")
        ax.set_ylabel(r"Free surface elevation $(m)$")
        ax.set_xlim( exp_data.time.iloc[0], exp_data.time.iloc[-1] )
        ax.legend()
        fig.savefig(os.path.join("results", "stage"), bbox_inches="tight")
        plt.close()
            
    def plot_speedups(
        self,
        my_rc_params
    ):
        plt.rcParams.update(my_rc_params)
        
        fig, ax = plt.subplots()
        
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
            "legend.fontsize" : "large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large"
        }
        
        self.plot_exp_data(my_rc_params, exp_data)
        
        self.plot_speedups(my_rc_params)
        
if __name__ == "__main__":
    subprocess.run( ["python", "stage.py" ] )
    subprocess.run( ["python", "inflow.py"] )
    subprocess.run( ["python", "raster.py"] )
    
    SimulationMonai( [1e-3, 1e-4, 0], ["mw"] ).plot( ExperimentalDataMonai() )