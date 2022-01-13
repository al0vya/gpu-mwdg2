import os
import subprocess
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

class CrossSectionData:
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
        
        self.xmin       = xmin
        self.cellsize   = cellsize
        self.epsilons   = epsilons
        self.i_range    = [ _ for _ in range(col_start,col_end) ]
        self.intervals  = [ _ for _ in range(4,9) ]
        self.velocities = {}
        
        for interval in self.intervals:        
            if interval not in self.velocities:
                self.velocities[interval] = {}
                for key in (*self.epsilons, "lisflood"):
                    if key not in self.velocities[interval]:
                        self.velocities[interval][key] = {}
        
        y_range = 0.2 - ymin
        
        row = int(y_range / cellsize)
        
        # reading from the top down
        # +6 to skip the 6 header lines
        self.slice_row = nrows - row + 6

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
                    "saveint     1\n" +
                    "sim_time    10\n" +
                    "raster_out  on\n" +
                    "solver      hw\n" +
                    "wall_height 2.5"
                ) % epsilon
                
                fp.write(params)
            
            subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "25-blocks.par"] )
        
    def extract_adaptive(
            self,
            epsilon
        ):
            for interval in self.intervals:
                extract_file = os.path.join("results", "results-" + str(interval) + ".wd")
                
                self.velocities[interval][epsilon] = np.loadtxt(
                    fname=extract_file,
                    skiprows=self.slice_row,
                    max_rows=1,
                    usecols=self.i_range
                )
                
    def run_and_extract_adaptive_all(self):
        for epsilon in self.epsilons:
            self.run_adaptive(epsilon)
            self.extract_adaptive(epsilon)
    
    def plot(self):
        fig, ax = plt.subplots()
        
        for interval in self.intervals:
            for config, data in self.velocities[interval].items():
                if config == "lisflood": continue
                ax.plot( [self.xmin + i * self.cellsize for i in self.i_range], data, label=config )  
            
            plt.legend()
            plt.savefig("why-" + str(interval))
            ax.clear()
            
    def run_extract_and_plot_all(self):
        self.run_and_extract_adaptive_all()
        self.plot()
    
if __name__ == "__main__":
    CrossSectionData( [1e-2, 1e-3] ).run_extract_and_plot_all()