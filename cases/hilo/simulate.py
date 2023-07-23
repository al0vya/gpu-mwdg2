import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "Use this tool as follows:\n" +
        "python simulate.py preprocess\n" +
        "python simulate.py simulate <SOLVER> <EPSILON> <RESULTS_DIRECTORY>"
    )
    
    sys.exit(help_message)
    
def check_raster_file(
    raster,
    xmin,
    ymin,
    cellsize,
    nrows,
    ncols,
    filename
):
    fig, ax = plt.subplots( figsize=(6,4) )
    
    y = [ ymin + j * cellsize for j in range(nrows) ]
    x = [ xmin + i * cellsize for i in range(ncols) ]
    
    aspect = ( y[-1] - y[0] ) / ( x[-1] - x[0] )
    
    x, y = np.meshgrid(x, y)
    
    contourset = ax.contourf(x, y, raster, levels=[-30 + i for i in range(30)])
    
    fig.colorbar(
        contourset,
        orientation="horizontal",
        label=r"$m$"
    )
    
    ax.legend()
    
    plt.setp(
        ax,
        xlabel=r"$x \, (m)$",
        ylabel=r"$y \, (m)$"
    )
    
    fig.tight_layout()
    
    fig.savefig(filename + ".png", bbox_inches="tight")
    
    plt.close()
    
def write_raster_file(
    raster,
    xmin,
    ymin,
    cellsize,
    nrows,
    ncols,
    filename
):
    print("Preparing raster file: %s..." % filename)
    
    check_raster_file(
        raster=raster,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols,
        filename=filename
    )
    
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
        str(xmin),
        str(ymin),
        str(cellsize)
    )
    
    np.savetxt(
        fname=filename,
        X=np.flipud(raster),
        fmt="%.8f",
        header=header,
        comments=""
    )
    
def write_all_input_files():
    print("Loading DEM...")
    
    nrows = 692
    ncols = 701
    
    bathymetry = np.loadtxt(
        fname=os.path.join("input-data", "hilo_grid_1_3_arcsec.txt"),
        usecols=2
    ).reshape( (nrows,ncols) )
    
    write_raster_file(
        raster=bathymetry,
        xmin=0,
        ymin=0,
        cellsize=1,
        nrows=nrows,
        ncols=ncols,
        filename="hilo.dem"
    )
    
write_all_input_files()