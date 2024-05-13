# script to run generate the input raster files for Monai valley simulation
# data obtained from https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html

import os
import sys
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

# wave enters from the west
def plot_impact_wave_direction(ax):
    xmax = 392 * 0.014
    ymax = 243 * 0.014
    
    num_arrows = 5
    
    gap_between_arrows = ymax / num_arrows
    
    arrow_centres = [0.5 * gap_between_arrows + j * gap_between_arrows for j in range(num_arrows)]
    
    for centre in arrow_centres:
        ax.arrow(x=0, y=centre, dx=4.6, dy=0, head_width=0.1, color='r')

def remove_NODATA_values(
        nodal_data,
        NODATA_value
    ):
        tol_0 = 1e-10
        
        for j, row in enumerate(nodal_data):
            for i, element in enumerate(row):
                if ( (element - NODATA_value) < tol_0 ): nodal_data[j, i] = 100
                
def check_nodal_data(
        nodal_data,
        nrows,
        ncols,
        xmin,
        ymin,
        cellsize,
        filename
    ):
        x = [ xmin + cellsize * i for i in range(ncols) ]
        y = [ ymin + cellsize * j for j in range(nrows) ]
        
        x, y = np.meshgrid(x, y)
        
        fig, ax    = plt.subplots( figsize=(5.0,4.2) )
        contourset = ax.contourf(x, y, nodal_data, levels=30)
        colorbar   = fig.colorbar(
            contourset,
            orientation="horizontal",
            label='m'
        )
        
        ax.scatter(4.501, 1.196, label="Point 1", color='c')
        ax.scatter(4.501, 1.696, label="Point 2", color='m')
        ax.scatter(4.501, 2.196, label="Point 3", color='y')
        
        plot_impact_wave_direction(ax)
        
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        
        ax.legend()
        
        fig.savefig(fname=filename + ".svg", bbox_inches="tight")

def projection(
        nodal_data
    ):
        raster = np.full(shape=(nodal_data.shape[0]-1, nodal_data.shape[1]-1), fill_value=-9999, dtype=float)
        
        tol_0 = 1e-10
        
        for j in range(nodal_data.shape[0]-1):
            for i in range(nodal_data.shape[1]-1):
                NE = nodal_data[j, i]
                NW = nodal_data[j, i+1]
                SE = nodal_data[j+1, i]
                SW = nodal_data[j+1, i+1]
                
                raster[j, i] = 0.25 * (NE + NW + SE + SW)
                
        return raster

def write_raster_file(
        filename,
        raster,
        nrows,
        ncols,
        xmin,
        ymin,
        cellsize
    ):
        header = (
            "ncols        %s\n" +
            "nrows        %s\n" +
            "xllcorner    %s\n" +
            "yllcorner    %s\n" +
            "cellsize     %s\n" +
            "NODATA_value -9999"
        ) % (
            ncols,
            nrows,
            xmin,
            ymin,
            cellsize
        )
        
        np.savetxt(filename, np.flipud(raster), fmt="%.15f", header=header, comments="")
   
def main():
    print("Preparing input raster files...")
    
    nrows = 243
    ncols = 392
    
    bed_data = np.loadtxt(fname="bed-data.txt", skiprows=1, usecols=2).reshape(ncols+1, nrows+1).transpose()
    
    # adjustment for datum
    bed_data *= -1
    bed_data +=  0.13535
    
    remove_NODATA_values(nodal_data=bed_data, NODATA_value=-30)
    
    cellsize = 0.014
    
    check_nodal_data(
        nodal_data=bed_data,
        nrows=nrows+1,
        ncols=ncols+1,
        xmin=0,
        ymin=0,
        cellsize=cellsize,
        filename="monai-topography"
    )
    
    # project from nodal to modal data
    bed_data = projection(nodal_data=bed_data)
    
    # upscale
    bed_data = np.kron(bed_data, np.ones((2,2)))
    
    initial_depths = np.maximum(0.13535 - bed_data, 0)
    
    write_raster_file(
        nrows=2*nrows,
        ncols=2*ncols,
        raster=bed_data,
        filename="monai.dem",
        xmin=0,
        ymin=0,
        cellsize=cellsize / 2
    )
    
    write_raster_file(
        nrows=2*nrows,
        ncols=2*ncols,
        raster=initial_depths,
        filename="monai.start",
        xmin=0,
        ymin=0,
        cellsize=cellsize / 2
    )
    
if __name__ == "__main__":
    main()