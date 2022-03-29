import os
import sys
import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

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
        print("Projecting and writing raster field:", filename)
        
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
   
def conical_island_raster_fields(x, y, x1, y1):
    print("Computing conical island input raster fields...")
    
    h    = 0
    h_2D = []
    h_1D = []
    
    qx    = 0
    qx_2D = []
    qx_1D = []
    
    z    = 0
    z_2D = []
    z_1D = []
    
    g  = 9.80665
    h0 = 0.32
    a  = 0.014
    c  = np.sqrt( g * (h0 + a) )
    k  = np.sqrt( 3 * a / (4 * (h0 + a) * h0 * h0) )
    
    for y_ in y:
        for x_ in x:
            r1 = np.sqrt( (x_ - x1) * (x_ - x1) + (y_ - y1) * (y_ - y1) )
            
            if r1 <= 1.1:
                z = 0.625
            elif r1 >= 1.1 and r1 <= 3.6:
                z = (3.6 - r1) / 4
            else:
                z = 0
            
            h = max(0, h0 - z)
            
            h_ = h0 + a * ( ( 1 / np.cosh(k * x_) ) * ( 1 / np.cosh(k * x_) ) )
            
            h = h_ - z
            
            qx = h * c * (1 - (h0 - z) / h)
            
            h_1D.append(h)
            qx_1D.append(qx)
            z_1D.append(z)
            
        h_2D.append(h_1D)
        qx_2D.append(qx_1D)
        z_2D.append(z_1D)
        
        h_1D  = []
        qx_1D = []
        z_1D  = []
    
    h_2D  = np.array(h_2D)
    qx_2D = np.array(qx_2D)
    z_2D  = np.array(z_2D)
    
    return {"h" : h_2D, "qx" : qx_2D, "z" : z_2D}

def main():
    start_file_exists   = os.path.exists("conical-island.start")
    startQx_file_exists = os.path.exists("conical-island.start.Qx")
    dem_file_exists     = os.path.exists("conical-island.dem")
    
    if start_file_exists and startQx_file_exists and dem_file_exists:
        sys.exit("All input raster files already exist, exiting raster file generation.")
    
    xmin = -5
    xmax = 26
    
    ymin = 0
    ymax = 27.6
    
    cellsize = 0.05;
    
    x = np.arange(xmin, xmax + cellsize, cellsize, dtype=float)
    y = np.arange(ymin, ymax + cellsize, cellsize, dtype=float)
    
    x1 = 12.96
    y1 = 13.80
    
    raster_fields = conical_island_raster_fields(x, y, x1, y1)
    
    nrows, ncols = raster_fields["h"].shape
    
    print("Showing raster fields for checking, close the plots to continue.")
    
    check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=raster_fields["h"])
    check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=raster_fields["qx"])
    check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=raster_fields["z"])
    
    if start_file_exists:
        print("Not writing conical-island.start because it already exists.")
    else:
        project_and_write_raster(
            nrows=nrows,
            ncols=ncols,
            nodal_data=raster_fields["h"],
            filename="conical-island.start",
            xmin=xmin,
            ymin=ymin,
            cellsize=cellsize
        )
    
    if startQx_file_exists:
        print("Not writing conical-island.start.Qx because it already exists.")
    else:
        project_and_write_raster(
            nrows=nrows,
            ncols=ncols,
            nodal_data=raster_fields["qx"],
            filename="conical-island.start.Qx",
            xmin=xmin,
            ymin=ymin,
            cellsize=cellsize
        )
    
    if dem_file_exists:
        print("Not writing conical-island.dem because it already exists.")
    else:
        project_and_write_raster(
            nrows=nrows,
            ncols=ncols,
            nodal_data=raster_fields["z"],
            filename="conical-island.dem",
            xmin=xmin,
            ymin=ymin,
            cellsize=cellsize
        )

if __name__ == "__main__":
    main()