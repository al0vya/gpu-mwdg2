import os
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class ExperimentalDataMonai:
    def __init__(self):
        experimental_dataframe = pd.read_csv("experimental.txt")
        
        self.time       = experimental_dataframe["time"]
        self.gauge_data = experimental_dataframe["stage1"]

class SimulationMonai:
    def __init__(
            self,
            epsilons
        ):
            self.configs      = (*epsilons, "lisflood")
            self.fields       = ["simtime", "runtime", "gauge_data"]
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "simtime-vs-runtime.csv")
            self.results      = {}
            
            for config in self.configs:
                if config not in self.results:
                    self.results[config] = {}
                    for field in self.fields:
                        if field not in self.results[config]:
                            self.results[config][field] = {}
                            
            for epsilon in epsilons:
                #self.run_adaptive(epsilon)
                
                time_dataframe = pd.read_csv(self.runtime_file)
                
                self.results[epsilon]["simtime"]    = time_dataframe["simtime"]
                self.results[epsilon]["runtime"]    = time_dataframe["runtime"]
                self.results[epsilon]["gauge_data"] = pd.read_csv(self.stage_file)["stage1"]
            
    def run_adaptive(
            self,
            epsilon
        ):
            with open("monai.par", 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 9\n" +
                    "min_dt      1\n" +
                    "respath     .\\results\n" +
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
                    "solver      mw\n" +
                    "cumulative  on\n" +
                    "wall_height 0.5"
                ) % epsilon
                
                fp.write(params)
            
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "monai.par"] )
            
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
            
            for config in self.configs:
                if config == "lisflood": continue
                
                ax.plot(
                    self.results[config]["simtime"],
                    self.results[config]["gauge_data"] - 0.13535,
                    linewidth=2.5,
                    label=config
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
            ax.clear()
            
            for config in self.configs:
                if config == "lisflood": continue
                
                runtime_ratio = self.results[config]["runtime"] / self.results[0]["runtime"]
                
                ax.plot(
                    self.results[config]["simtime"],
                    runtime_ratio,
                    linewidth=2.5,
                    label=config
                )
            
            config = self.configs[0]
            
            xlim = (
                ( self.results[config]["simtime"] ).iloc[0],
                ( self.results[config]["simtime"] ).iloc[-1]
            )
            
            ax.set_xlabel(r"$t \, (s)$")
            ax.set_ylabel("Speedup ratio GPU-MWDG2/GPU-DG2")
            ax.set_xlim(xlim)
            ax.legend()
            fig.savefig(os.path.join("results", "runtimes"), bbox_inches="tight")
            ax.clear()
            
            plt.close()
        
if __name__ == "__main__":
    SimulationMonai( [1e-3, 1e-4, 0] ).plot( ExperimentalDataMonai() )