import os
import subprocess
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

class ExperimentalData:
    def __init__(self):
        depth_exp_df_4s  = pd.read_csv("WLxy_C1_4s.txt",  header=None, delimiter=' ')
        depth_exp_df_5s  = pd.read_csv("WLxy_C1_5s.txt",  header=None, delimiter=' ')
        depth_exp_df_6s  = pd.read_csv("WLxy_C1_6s.txt",  header=None, delimiter=' ')
        depth_exp_df_10s = pd.read_csv("WLxy_C1_10s.txt", header=None, delimiter=' ')

        velocity_exp_df_4s  = pd.read_csv("Vxy_C1_4s.txt",  header=None, delimiter=' ')
        velocity_exp_df_5s  = pd.read_csv("Vxy_C1_5s.txt",  header=None, delimiter=' ')
        velocity_exp_df_6s  = pd.read_csv("Vxy_C1_6s.txt",  header=None, delimiter=' ')
        velocity_exp_df_10s = pd.read_csv("Vxy_C1_10s.txt", header=None, delimiter=' ')
        
        self.data = {}
        
        intervals = [4, 5, 6, 10]
        flow_vars = ["depth", "velocity"]
        
class Simulation25Blocks:
    def __init__(
            self,
            epsilons
        ):
            header = []
            
            with open("25-blocks.start", 'r') as fp:
                for i, line in enumerate(fp):
                    if i > 4: break
                    header.append(line)
            
            cast = lambda ans, i : int(ans) if i < 2 else float(ans)
            
            ncols, nrows, xmin, ymin, cellsize = [ cast(line.split()[1], i) for i, line in enumerate(header) ]
            
            x_range_start = 4 - xmin
            x_range_end   = 8 - xmin
            
            col_start = int(x_range_start / cellsize)
            col_end   = int(x_range_end   / cellsize)
            
            self.xmin         = xmin
            self.cellsize     = cellsize
            self.runtime_file = os.path.join("results", "simtime-vs-runtime.csv")
            self.configs      = (*epsilons, "lisflood")
            self.flow_vars    = ("depth", "velocity")
            self.time_vars    = ("simtime", "runtime")
            self.i_range      = [ _ for _ in range(col_start,col_end) ]
            self.intervals    = [4, 5, 6, 10]
            self.results      = {}
            
            for config in self.configs:        
                if config not in self.results:
                    self.results[config] = {}
                    for var in self.time_vars:
                        if var not in self.results[config]:
                            self.results[config][var] = {}
                    for interval in self.intervals:
                        if interval not in self.results[config]:
                            self.results[config][interval] = {}
                            for var in self.flow_vars:
                                if var not in self.results[config][interval]:
                                    self.results[config][interval][var] = {}
            
            y_range = 0.2 - ymin
            
            row = int(y_range / cellsize)
            
            # reading from the top down
            # +6 to skip the 6 header lines
            self.slice_row = nrows - row + 6
            
            for epsilon in epsilons:
                self.run_adaptive(epsilon)
                
                time_dataframe = pd.read_csv(self.runtime_file)
                
                self.results[epsilon]["simtime"] = time_dataframe["simtime"]
                self.results[epsilon]["runtime"] = time_dataframe["runtime"]
                
                for interval in self.intervals:
                    h_raster_file  = os.path.join("results", "results-" + str(interval) + ".wd")
                    qx_raster_file = os.path.join("results", "results-" + str(interval) + ".qx")
                    
                    self.results[epsilon][interval]["depth"] = np.loadtxt(
                        fname=h_raster_file,
                        skiprows=self.slice_row,
                        max_rows=1,
                        usecols=self.i_range
                    )
                    
                    # calculating velocity from depth and discharge
                    self.results[epsilon][interval]["velocity"] = np.loadtxt(
                        fname=qx_raster_file,
                        skiprows=self.slice_row,
                        max_rows=1,
                        usecols=self.i_range
                    ) / self.results[epsilon][interval]["depth"]
                        
    def run_adaptive(
            self,
            epsilon
        ):
            with open("25-blocks.par", 'w') as fp:
                params = (
                    "test_case   0\n" +
                    "max_ref_lvl 11\n" +
                    "min_dt      1\n" +
                    "respath     .\\results\n" +
                    "epsilon     %s\n" +
                    "fpfric      0.01\n" +
                    "rasterroot  25-blocks\n" +
                    "tol_h       1e-3\n" +
                    "tol_q       0\n" +
                    "tol_s       1e-9\n" +
                    "g           9.80665\n" +
                    "massint     0.1\n" +
                    "saveint     1\n" +
                    "sim_time    10\n" +
                    "cumulative  on\n" +
                    "raster_out  on\n" +
                    "solver      hw\n" +
                    "wall_height 2.5"
                ) % epsilon
                
                fp.write(params)
                
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "25-blocks.par"] )
            
    def plot(self):
        fig, ax = plt.subplots()
        
        x = [self.xmin + i * self.cellsize for i in self.i_range]
        
        for interval in self.intervals:
            for config in self.configs:
                if config == "lisflood": continue
                ax.plot(x, self.results[config][interval]["depth"], label=config)
            
            plt.legend()
            plt.savefig( os.path.join("results", "depth-" + str(interval) ) )
            ax.clear()
            
            for config in self.configs:
                if config == "lisflood": continue
                ax.plot(x, self.results[config][interval]["velocity"], label=config)
            
            plt.legend()
            plt.savefig( os.path.join("results", "velocity-" + str(interval) ) )
            ax.clear()
            
        for config in self.configs:
            if config == "lisflood": continue
            runtime_ratio = self.results[config]["runtime"] / self.results[1e-2]["runtime"]
            ax.plot(self.results[config]["simtime"], runtime_ratio, label=config)
        
        plt.legend()
        plt.savefig( os.path.join("results", "runtimes") )
        ax.clear()
        
if __name__ == "__main__":
    Simulation25Blocks( [1e-2, 1e-3] ).plot()