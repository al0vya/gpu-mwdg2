import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def write_bdy_file(
    timeseries_name,
    datum
):
    print("Preparing bdy file...")
    
    boundary_timeseries = np.loadtxt( fname=os.path.join("input-data", "ts_5m.txt") )
    
    fig, ax = plt.subplots()
    
    ax.plot(boundary_timeseries[:,0], boundary_timeseries[:,1])
    
    plt.setp(
        ax,
        title="Western boundary timeseries",
        xlim=( boundary_timeseries[0,0], boundary_timeseries[-1,0] ),
        xlabel=r"$t \, (s)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    fig.savefig(os.path.join("results", "input-wave.png"), bbox_inches="tight")
    
    plt.close()
    
    timeseries_len = boundary_timeseries.shape[0]
    
    with open("seaside-oregon.bdy", 'w') as fp:
        header = (
            "%s\n" +
            "%s seconds\n"
        ) % (
            timeseries_name,
            timeseries_len,
        )
        
        fp.write(header)
        
        for entry in boundary_timeseries:
            fp.write(str(entry[1] - datum) + " " + str( entry[0] ) + "\n")
    
def write_dem_file(
    bathymetry,
    datum,
    xmin,
    ymin,
    cellsize,
    nrows,
    ncols
):
    print("Preparing DEM file...")
    
    fig, ax = plt.subplots()
    
    y = [ ymin + j * cellsize for j in range(nrows) ]
    x = [ xmin + i * cellsize for i in range(ncols) ]
    
    x, y = np.meshgrid(x, y)
    
    adjusted_bathymetry = bathymetry - datum
    
    contourset = ax.contourf(x, y, adjusted_bathymetry)
    
    fig.colorbar(contourset)
    
    plt.setp(
        ax,
        xlabel=r"$x \, (m)$",
        ylabel=r"$y \, (m)$"
    )
    
    fig.savefig(os.path.join("results", "bathymetry.png"), bbox_inches="tight")
    
    plt.close()
    
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
        fname="seaside-oregon-0.01m.dem",
        X=np.flipud(adjusted_bathymetry),
        fmt="%.8f",
        header=header,
        comments=""
    )

def write_bci_file(
    ymin,
    nrows,
    cellsize,
    timeseries_name
):
    print("Preparing bci file...")
    
    ymax = ymin + nrows * cellsize
    
    with open("seaside-oregon.bci", 'w') as fp:
        fp.write( "W %s %s HVAR %s" % (ymin, ymax, timeseries_name) )
    

def write_all_input_files():
    print("Loading DEM...")
    
    bathymetry = np.loadtxt( fname=os.path.join("input-data", "bathymetry.csv"), delimiter="," );
    
    datum = bathymetry.min()
    
    nrows, ncols = bathymetry.shape
    
    # obtained by checking difference between items in x array after running input-data/plot_bathy.m
    cellsize = 0.01
    
    # numerical values obtained by checking the first item of the x and y arrays after running input-data/plot_bathy.m
    xmin =   0.0120010375976563 - cellsize / 2
    ymin = -13.2609996795654    - cellsize / 2
    
    write_dem_file(
        bathymetry=bathymetry,
        datum=datum,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols
    )
    
    timeseries_name = "INLET"
    
    write_bdy_file(
        timeseries_name=timeseries_name,
        datum=datum
    )
    
    write_bci_file(
        ymin=ymin,
        nrows=nrows,
        cellsize=cellsize,
        timeseries_name=timeseries_name
    )
    
def main():
    write_all_input_files()

if __name__ == "__main__":
    main()