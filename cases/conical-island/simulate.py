import os
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class ExperimentalDataConicalIsland:
    def __init__(self):
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
            epsilons
        ):
            self.configs      = (*epsilons, "lisflood")
            self.fields       = ["simtime", "runtime", "gauge_data"]
            
            # stages 6, 9, 12 and 22 from
            # "Laboratory experiments of tsunami runup on a circular island"
            self.stages       = [ "#" + str(i) for i in [6, 9, 16, 22] ] 
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "simtime-vs-runtime.csv")
            self.results      = {}
            
            for config in self.configs:
                self.results[config] = {}

                for field in self.fields:
                    self.results[config][field] = {}
                    
                for stage in self.stages:
                    self.results[config]["gauge_data"][stage] = {}
                    
            for epsilon in epsilons:
                self.run_adaptive(epsilon)
                
                time_dataframe = pd.read_csv(self.runtime_file)
                
                self.results[epsilon]["simtime"] = time_dataframe["simtime"]
                self.results[epsilon]["runtime"] = time_dataframe["runtime"]
                
                stage_dataframe = pd.read_csv(self.stage_file)
                
                for i, stage in enumerate(self.stages):
                    self.results[epsilon]["gauge_data"][stage] = stage_dataframe.iloc[:,i+1]
                    
    def run_adaptive(
            self,
            epsilon
        ):
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
                    "solver      mw\n" +
                    "cumulative  on\n" +
                    "wall_height 1"
                ) % epsilon
                
                fp.write(params)
            
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "conical-island.par"] )
            
    def plot(
            self,
            exp_data
        ):
            my_rc_params = {
                "legend.fontsize" : "xx-large",
                "axes.labelsize"  : "xx-large",
                "axes.titlesize"  : "xx-large",
                "xtick.labelsize" : "xx-large",
                "ytick.labelsize" : "xx-large"
            }
            
            plt.rcParams.update(my_rc_params)
            
            fig, ax = plt.subplots()
            
            for stage in self.stages:
                for config in self.configs:
                    if config == "lisflood": continue
                    
                    ax.plot(
                        self.results[config]["simtime"],
                        self.results[config]["gauge_data"][stage],
                        linewidth=2.5,
                        label=config
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
                #ax.set_xlim( exp_data.data["time"][stage].iloc[0], exp_data.data["time"][stage].iloc[-1] )
                ax.legend()
                fig.savefig(os.path.join("results", "stage-" + stage), bbox_inches="tight")
                ax.clear()
            
            for config in self.configs:
                if config == "lisflood": continue
                
                runtime_ratio = self.results[0]["runtime"] / self.results[config]["runtime"]
                
                ax.plot(
                    self.results[config]["simtime"],
                    runtime_ratio,
                    linewidth=2.5,
                    label=config
                )
            
            xlim = (
                ( self.results[0]["simtime"] ).iloc[0],
                ( self.results[0]["simtime"] ).iloc[-1]
            )
            
            ax.set_xlabel(r"$t \, (s)$")
            ax.set_ylabel("Speedup ratio GPU-MWDG2/GPU-DG2")
            ax.set_xlim(xlim)
            ax.legend()
            fig.savefig(os.path.join("results", "runtimes"), bbox_inches="tight")
            ax.clear()
            
            plt.close()
        
if __name__ == "__main__":
    with open("conical-island.stage", 'w') as fp:
        stages = (
            "4\n" +
            "9.36  13.8\n" +
            "10.36 13.8\n" +
            "12.96 11.22\n" +
            "15.56 13.8"
        )
        
        fp.write(stages)
    
    SimulationConicalIsland( [1e-3, 1e-4, 0] ).plot( ExperimentalDataConicalIsland() )