import pandas            as pd
import numpy             as np
import matplotlib.pyplot as plt
 
from mpl_toolkits.mplot3d import Axes3D

def check_nodal_data(
        nodal_data,
        nrows=501,
        ncols=901
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
        NODATA_value,
        nrows=501,
        ncols=901
    ):
        raster = np.full(shape=(nrows-1, ncols-1), fill_value=-9999, dtype=float)
        
        tol_0 = 1e-10
        
        for j in range(nrows-1):
            for i in range(ncols-1):
                NE = nodal_data[j, i]
                NW = nodal_data[j, i+1]
                SE = nodal_data[j+1, i]
                SW = nodal_data[j+1, i+1]
                
                not_NODATA = (
                    not( (NE - NODATA_value) < tol_0 ) and
                    not( (NW - NODATA_value) < tol_0 ) and
                    not( (SE - NODATA_value) < tol_0 ) and
                    not( (SW - NODATA_value) < tol_0 )
                )
                
                if not_NODATA: raster[j, i] = 0.25 * (NE + NW + SE + SW)
                
        return raster

def write_raster_file(
        filename,
        raster,
        nrows=501,
        ncols=901
    ):
        header = (
            "ncols        %s\n" +
            "nrows        %s\n" +
            "xllcorner    0\n" +
            "yllcorner    0\n" +
            "cellsize     20\n" +
            "NODATA_value -9999"
        ) % (
            ncols-1,
            nrows-1
        )
        
        np.savetxt(filename, raster, fmt="%.15f", header=header, comments="")

def project_and_write_raster(
        nodal_data,
        NODATA_value,
        filename,
        nrows=501,
        ncols=901
    ):
        raster = projection(
            nodal_data=nodal_data,
            NODATA_value=NODATA_value
        )
        
        write_raster_file(
            raster=raster,
            filename=filename
        )

xmin = -5
xmax = 26

ymin = 0
ymax = 27

cell_size = 0.05;

x = np.arange(xmin, xmax + cell_size, 0.05, dtype=float)
y = np.arange(ymin, ymax + cell_size, 0.05, dtype=float)

x1 = 12.96
y1 = 13.80

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
        elif r1 > 1.1 and r1 <= 3.6:
            z = (3 - r1) / 4
        else:
            z = 0
        
        z = max(0, z)
        
        h = h0 + a * ( ( 1 / np.cosh(k * x_) ) * ( 1 / np.cosh(k * x_) ) ) - z
        
        qx = h * c * (1 - (h0 - z) / h)
        
        h = max(0, h)
        
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

X, Y = np.meshgrid(x, y)

fig, ax = plt.subplots( subplot_kw={"projection" : "3d"} )

ax.plot_surface(X, Y,  h_2D); plt.show()
ax.plot_surface(X, Y, qx_2D); plt.show()
ax.plot_surface(X, Y,  z_2D); plt.show()

nrows, ncols = *h_2D.shape[0]

project_and_write_raster(
    nrows=nrows,
    ncols=ncols,
    nodal_data=h_2D,
    NODATA_value=-30,
    filename="conical-island.start"
)

project_and_write_raster(
    nrows=nrows,
    ncols=ncols,
    nodal_data=qx_2D,
    NODATA_value=-30,
    filename="conical-island.startQx"
)

project_and_write_raster(
    nrows=nrows,
    ncols=ncols,
    nodal_data=z_2D,
    NODATA_value=-30,
    filename="conical-island.dem"
)

