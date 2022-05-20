import os
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

class SimulationMalpasset:
    def __init__(
            self,
            epsilons
        ):
            self.configs      = (*epsilons, "lisflood")
            self.fields       = ["simtime", "runtime", "arrival_time"]
            self.stages       = [ _ for _ in range(1,10) ]
            self.stage_file   = os.path.join("results", "stage.wd")
            self.runtime_file = os.path.join("results", "cumulative-data.csv")
            self.results      = {}
            
            for config in self.configs:
                self.results[config] = {}
                
                for field in self.fields:
                    self.results[config][field] = {}
                
                self.results[config]["arrival_time"] = [0 for stage in self.stages]
                    
            for epsilon in epsilons:
                self.run_adaptive(epsilon)
                
                time_dataframe = pd.read_csv(self.runtime_file)
                
                self.results[epsilon]["simtime"] = time_dataframe["simtime"]
                self.results[epsilon]["runtime"] = time_dataframe["runtime"]
                
                with open(self.stage_file, 'r') as fp:
                    for col, stage in enumerate(self.stages):
                        for row, line in enumerate(fp):
                            if row == 0: continue
                            
                            items = line.split(',')
                            time  = float( items[0] )
                            depth = float( items[stage] )
                            
                            if depth > 0:
                                self.results[epsilon]["arrival_time"][col] = time / 60
                                break
                                
    def run_adaptive(
            self,
            epsilon
        ):
            with open("malpasset.par", 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 10\n" +
                    "min_dt      1\n" +
                    "respath     results\n" +
                    "epsilon     %s\n" +
                    "fpfric      0.033\n" +
                    "rasterroot  malpasset\n" +
                    "stagefile   malpasset.stage\n" +
                    "tol_h       1e-3\n" +
                    "tol_q       0\n" +
                    "tol_s       1e-9\n" +
                    "g           9.80665\n" +
                    "saveint     100\n" +
                    "massint     1\n" +
                    "sim_time    600\n" +
                    "solver      mw\n" +
                    "vtk         on\n" +
                    "cumulative  on\n" +
                    "wall_height 100"
                ) % epsilon
                
                fp.write(params)
            
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "malpasset.par"] )
            
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
                    self.stages,
                    self.results[config]["arrival_time"],
                    linewidth=2.5,
                    label=config
                )
            
            ax.scatter(
                self.stages,
                exp_data,
                facecolor="None",
                edgecolor="black",
                label="Experimental"
            )
            
            ax.set_xlabel("Stage")
            ax.set_ylabel(r"Arrival time $(s)$")
            ax.set_xlim( self.stages[0], self.stages[-1] )
            ax.legend()
            fig.savefig(os.path.join("results", "arrival-times"), bbox_inches="tight")
            ax.clear()
            
            for config in self.configs:
                if config == "lisflood": continue
                
                runtime_ratio = self.results[1e-2]["runtime"] / self.results[config]["runtime"]
                
                ax.plot(
                    self.results[config]["simtime"],
                    runtime_ratio,
                    linewidth=2.5,
                    label=config
                )
            
            xlim = (
                ( self.results[1e-2]["simtime"] ).iloc[0],
                ( self.results[1e-2]["simtime"] ).iloc[-1]
            )
            
            ax.set_xlabel(r"$t \, (s)$")
            ax.set_ylabel("Speedup ratio GPU-MWDG2/GPU-DG2")
            ax.set_xlim(xlim)
            ax.legend()
            fig.savefig(os.path.join("results", "runtimes"), bbox_inches="tight")
            ax.clear()
            
            plt.close()
        
if __name__ == "__main__":
    with open("malpasset.stage", 'w') as fp:
        stages = (
            "9\n" +
            "4947.4  7289.7\n"
            "5717.3  7407.6\n"
            "6775.1  6869.2\n"
            "7128.2  6162.0\n"
            "8585.3  6443.1\n"
            "9675.0  6085.9\n"
            "10939.1 6044.8\n"
            "11724.4 5810.4\n"
            "12723.7 5485.1"
        )
        
        fp.write(stages)
    
    exp_data = [
        0.2152,
        1.7683,
        3.0395,
        3.4321,
        6.7081,
        10.0126,
        14.0366,
        16.0899,
        18.8940
    ]
    
    SimulationMalpasset( [1e-2] ).plot(exp_data)