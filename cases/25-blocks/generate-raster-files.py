import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

def check_nodal_data(
        nodal_data,
        nrows,
        ncols
    ):
        x = np.linspace(0, 1, ncols)
        y = np.linspace(0, 1, nrows)
        
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots()
        ax.contourf(X, Y, nodal_data)
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
                NW = nodal_data[j, i]
                SE = nodal_data[j, i]
                SW = nodal_data[j, i]
                
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
        
        np.savetxt(filename, raster, fmt="%.15f", header=header, comments="")

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
        
def bed(x, y):
    def blocks(x, y):
        block = lambda x_lower, x_upper, y_lower, y_upper, x, y: \
            2 if x >= x_lower and x <= x_upper and y >= y_lower and y <= y_upper else 0
            
        x_bounds = [
            (5.00, 5.30),
            (5.40, 5.70),
            (5.80, 6.10),
            (6.20, 6.50),
            (6.60, 6.90)
        ]
        
        y_bounds = [
            (-0.95, -0.65),
            (-0.55, -0.25),
            (-0.15,  0.15),
            ( 0.25,  0.55),
            ( 0.65,  0.95)
        ]
        
        z = 0
        
        for y_bound in y_bounds:
            for x_bound in x_bounds:
                z = block(
                    x_lower=x_bound[0],
                    x_upper=x_bound[1],
                    y_lower=y_bound[0],
                    y_upper=y_bound[1],
                    x=x, y=y
                )
                
                if z > 0: break
                
            if z > 0: break
            
        return z
    
    z = 0
    
    # wall slopes
    slope     = 1.55 / 0.34
    intercept = 1.46
    
    if y >= 1.46 and y <= 1.9:
        z = slope * (y - intercept)
    elif y <= -1.46:
        z = slope * (-y - intercept)
        
    # gate walls
    if x >= 0 and x <= 0.8:
        if y <= -0.5 or y >= 0.5:
            z = 2
            
    # block obstacles
    z = max( z, blocks(x, y) )
    
    return z
            
def main():
    ncols    =  180 + 1
    nrows    =  180  + 1
    xmin     = -6.75
    ymin     = -1.80
    cellsize =  0.02
    
    x = [ xmin + i * cellsize for i in range(0, ncols) ]
    y = [ ymin + j * cellsize for j in range(0, nrows) ]
    
    bed_data   = np.full(shape=(nrows, ncols), fill_value=-9999, dtype=float)
    start_data = np.full(shape=(nrows, ncols), fill_value=0,     dtype=float)
    
    for j, y_ in enumerate(y):
        for i, x_ in enumerate(x):
            z = bed(x_, y_)
            
            bed_data[j, i]   = z
            start_data[j, i] = max(0, 0.011 - z) if x_ >= 0 else max(0, 0.011 - z)
            
    check_nodal_data(nodal_data=bed_data,   nrows=nrows, ncols=ncols)
    check_nodal_data(nodal_data=start_data, nrows=nrows, ncols=ncols)
    
    project_and_write_raster(
        nodal_data=bed_data,
        filename="25-blocks.dem",
        nrows=nrows,
        ncols=ncols,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize
    )
    
    project_and_write_raster(
        nodal_data=start_data,
        filename="25-blocks.start",
        nrows=nrows,
        ncols=ncols,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize
    )
            
if __name__ == "__main__":
    main()