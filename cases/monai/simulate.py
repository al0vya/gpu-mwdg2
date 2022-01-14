import os
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pylab  as pylab
import matplotlib.pyplot as plt

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
                self.run_adaptive(epsilon)
                
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
            
    def plot(self):
        params = {
            "legend.fontsize" : "xx-large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large"
        }
        
        pylab.rcParams.update(params)
        
        fig, ax = plt.subplots()
        
        for config in self.configs:
            if config == "lisflood": continue
            ax.plot(self.results[config]["simtime"], self.results[config]["gauge_data"], label=config)
        
        plt.legend()
        plt.savefig( os.path.join("results", "stage") )
        ax.clear()
        
        for config in self.configs:
            if config == "lisflood": continue
            runtime_ratio = self.results[config]["runtime"] / self.results[0]["runtime"]
            ax.plot(self.results[config]["simtime"], runtime_ratio, label=config)
        
        plt.legend()
        plt.savefig( os.path.join("results", "runtimes") )
        ax.clear()
        
if __name__ == "__main__":
    SimulationMonai( [1e-3, 1e-4, 0] ).plot()