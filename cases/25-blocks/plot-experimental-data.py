import os
import subprocess
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

class CrossSectionData:
    def __init__(self):
        header = []
        
        with open("25-blocks.start", 'r') as fp:
            for i, line in enumerate(fp):
                if i > 4: break
                header.append(line)
        
        cast = lambda ans, i : int(ans) if i < 2 else float(ans)
        
        ncols, nrows, xmin, ymin, cellsize = [ cast(line.split()[1], i) for i, line in enumerate(header) ]
        
        y_range = 0.2 - ymin
        
        row = int(y_range / cellsize)
        
        # reading from the top down
        # +6 to skip the 6 header lines
        row_read = nrows - row + 6
        
        x_range_start = 4 - xmin
        x_range_end   = 8 - xmin
        
        col_start = int(x_range_start / cellsize)
        col_end   = int(x_range_end   / cellsize)
        
        cols = [ _ for _ in range(col_start,col_end) ]
        
        self.x         = [xmin + i * cellsize for i in cols]
        self.intervals = [ _ for _ in range(4,9) ]
        
        self.velocities = {}

    def run_solver(epsilon):
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
        
    def extract(
            self,
            epsilon
        ):
            for interval in self.intervals:
                extract_file = os.path.join("results", "results-" + str(interval) + ".wd")
                save_file    = "mwdg2-eps-" + str(epsilon) + "-" + str(interval) + ".wd"
                
                self.velocities[interval] = np.loadtxt(
                    fname=extract_file,
                    skiprows=row_read,
                    max_rows=1,
                    usecols=cols
                )
    
    def plot(
            self,
            epsilon
        ):
            for interval in self.intervals:
                plt.plot(self.velocities[interval])
                plt.show()
                plt.close()
            
    def run_extract_and_plot(
            self,
            epsilon
        ):
            self.run_solver(self)
            self.extract(self, epsilon)
            self.plot(self, epsilon)
    
if __name__ == "__main__":
    CrossSectionData().run_extract_and_plot(1e-3)