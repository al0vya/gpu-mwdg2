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

def init_water_depths(
        bed_data,
        nrows,
        ncols,
        cellsize
    ):
        x = [ i * cellsize for i in range(ncols) ]
        y = [ j * cellsize for j in range(nrows) ]
        
        # dam as equation of straight line
        x1 = 4481
        y1 = 7344
        
        x2 = 4596
        y2 = 7118
        
        slope = (y2 - y1) / (x2 - x1)
        
        intercept = y1 - slope * x1
        
        dam = lambda x : slope * x + intercept
        
        h = np.full(shape=(nrows, ncols), fill_value=0, dtype=float)
        
        eta = 100
        
        for j, y_ in enumerate(y):
            for i, x_ in enumerate(x): 
                if y_ <= dam(x_): h[j, i] = max( 0, eta - bed_data[j, i] )
                    
        return h

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
        nrows = 501
        ncols = 901
        
        bed_data = np.loadtxt(fname="bed-data.txt", usecols=2).reshape(ncols, nrows).transpose()
        
        remove_NODATA_values(nodal_data=bed_data, NODATA_value=-30)
        
        cellsize = 20
        
        h = init_water_depths(nrows=nrows, ncols=ncols, bed_data=bed_data, cellsize=cellsize)
        
        check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=bed_data)
        check_nodal_data(nrows=nrows, ncols=ncols, nodal_data=h)
        
        project_and_write_raster(
            nrows=nrows,
            ncols=ncols,
            nodal_data=bed_data,
            filename="malpasset.dem",
            xmin=0,
            ymin=0,
            cellsize=cellsize
        )
        
        project_and_write_raster(
            nrows=nrows,
            ncols=ncols,
            nodal_data=h,
            filename="malpasset.start",
            xmin=0,
            ymin=0,
            cellsize=cellsize
        )
    
if __name__ == "__main__":
    main()