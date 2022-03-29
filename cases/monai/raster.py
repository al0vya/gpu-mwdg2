# script to run generate the input raster files for Monai valley simulation
# data obtained from https://nctr.pmel.noaa.gov/benchmark/Laboratory/Laboratory_MonaiValley/index.html

import sys
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

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
        ncols
    ):
        x = np.linspace(0, 1, ncols)
        y = np.linspace(0, 1, nrows)
        
        X, Y = np.meshgrid(x, y)
        
        fig, ax    = plt.subplots()
        contourset = ax.contourf(X, Y, nodal_data)
        colorbar   = fig.colorbar(contourset)
        
        plt.show()
        plt.close()

def projection(
        nodal_data,
        nrows,
        ncols
    ):
        raster = np.full(shape=(nrows-1, ncols-1), fill_value=-9999, dtype=float)
        
        tol_0 = 1e-10
        
        for j in range(nrows-1):
            for i in range(ncols-1):
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
            ncols-1,
            nrows-1,
            xmin,
            ymin,
            cellsize
        )
        
        np.savetxt(filename, np.flipud(raster), fmt="%.15f", header=header, comments="")

def project_and_write_raster(
        nodal_data,
        filename,
        nrows,
        ncols,
        xmin,
        ymin,
        cellsize
    ):
        raster = projection(
            nrows=nrows,
            ncols=ncols,
            nodal_data=nodal_data
        )
        
        write_raster_file(
            nrows=nrows,
            ncols=ncols,
            raster=raster,
            filename=filename,
            xmin=xmin,
            ymin=ymin,
            cellsize=cellsize
        )
   
def main():
    print("Preparing input raster files...")
    
    nrows = 243 + 1
    ncols = 392 + 1
    
    bed_data = np.loadtxt(fname="bed-data.txt", skiprows=1, usecols=2).reshape(ncols, nrows).transpose()
    
    # adjustment for datum
    bed_data *= -1
    bed_data +=  0.13535
    
    initial_depths = np.maximum(0.13535 - bed_data, 0)
    
    remove_NODATA_values(nodal_data=bed_data, NODATA_value=-30)
    
    cellsize = 0.014
    
    check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=bed_data)
    check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=initial_depths)
    
    project_and_write_raster(
        nrows=nrows,
        ncols=ncols,
        nodal_data=bed_data,
        filename="monai.dem",
        xmin=0,
        ymin=0,
        cellsize=cellsize
    )
    
    project_and_write_raster(
        nrows=nrows,
        ncols=ncols,
        nodal_data=initial_depths,
        filename="monai.start",
        xmin=0,
        ymin=0,
        cellsize=cellsize
    )
    
if __name__ == "__main__":
    main()