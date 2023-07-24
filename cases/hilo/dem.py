import os
import numpy as np
import matplotlib.pyplot as plt

def check_raster_file(
    raster,
    xmin,
    ymin,
    cellsize,
    nrows,
    ncols,
    filename
):
    fig, ax = plt.subplots( figsize=(4,4) )
    
    y = [ ymin + j * cellsize for j in range(nrows) ]
    x = [ xmin + i * cellsize for i in range(ncols) ]
    
    x, y = np.meshgrid(x, y)
    
    contourset = ax.contourf(x, y, raster, levels=30)
    
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

def write_dem_file():
    print("Generating DEM file...")
    
    filename = "hilo.dem"
    
    ncols    = 701
    nrows    = 692
    xmin     = 0
    ymin     = 0
    cellsize = 10
    
    dem = np.loadtxt(
        fname=os.path.join("input-data", "hilo_grid_1_3_arcsec.txt"),
        usecols=2
    ).reshape( (nrows,ncols) )
    
    # multiply by -1 to flip elevation
    dem *= -1
    
    dem = np.maximum( dem, np.full(shape=(nrows,ncols), fill_value=-30) )
    
    dem += 30
    
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
        X=np.flipud(dem),
        fmt="%.8f",
        header=header,
        comments=""
    )
    
    check_raster_file(
        raster=dem,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols,
        filename=filename
    )
    
def main():
    write_dem_file()

if __name__ == "__main__":
    main()