import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "Use this tool as follows:\n" +
        "python simulate.py preprocess\n" +
        "python simulate.py simulate <SOLVER> <EPSILON>"
    )
    
    sys.exit(help_message)

def write_bci_file(
    ymin,
    nrows,
    cellsize,
    timeseries_name
):
    print("Preparing bci file...")
    
    ymax = ymin + nrows * cellsize
    
    with open("oregon-seaside.bci", 'w') as fp:
        fp.write( "W %s %s HVAR %s" % (ymin, ymax, timeseries_name) )
    
def write_bdy_file(
    timeseries_name,
    datum
):
    print("Preparing bdy file...")
    
    boundary_timeseries = np.loadtxt( fname=os.path.join("input-data", "ts_5m.txt") )
    
    fig, ax = plt.subplots()
    
    ax.plot(
        boundary_timeseries[:,0],
        boundary_timeseries[:,1] - datum
    )
    
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
    
    with open("oregon-seaside.bdy", 'w') as fp:
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

def check_raster_file(
    raster,
    xmin,
    ymin,
    cellsize,
    nrows,
    ncols,
    filename
):
    fig, ax = plt.subplots()
    
    y = [ ymin + j * cellsize for j in range(nrows) ]
    x = [ xmin + i * cellsize for i in range(ncols) ]
    
    x, y = np.meshgrid(x, y)
    
    contourset = ax.contourf(x, y, raster)
    
    fig.colorbar(contourset)
    
    plt.setp(
        ax,
        xlabel=r"$x \, (m)$",
        ylabel=r"$y \, (m)$"
    )
    
    fig.savefig(os.path.join("results", filename + ".png"), bbox_inches="tight")
    
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

def downscale_raster(
    raster
):
    nrows, ncols = raster.shape
    
    # don't consider the last row/column if odd number of rows/columns
    x_lim = None if ncols % 2 == 0 else -1
    y_lim = None if nrows % 2 == 0 else -1
    
    top_left     = raster[ :y_lim:2,  :x_lim:2]
    top_right    = raster[ :y_lim:2, 1:x_lim:2]
    bottom_left  = raster[1:y_lim:2,  :x_lim:2]
    bottom_right = raster[1:y_lim:2, 1:x_lim:2]
    
    return (top_left + top_right + bottom_left + bottom_right) / 4
    
def write_stage_file():
    print("Preparing stage file...")
    
    stages = (
        "5\n"
        "18.618  0.000\n" + # W3
        "33.721 -0.588\n" + # B1
        "35.176 -0.406\n" + # B4
        "36.635 -0.229\n" + # B6
        "40.668  0.269\n"   # B9
    )
    
    with open("oregon-seaside.stage", 'w') as fp:
        fp.write(stages)

def write_all_input_files():
    print("Loading DEM...")
    
    bathymetry = np.loadtxt( fname=os.path.join("input-data", "bathymetry.csv"), delimiter="," );
    
    datum = bathymetry.min()
    
    adjusted_bathymetry            = bathymetry - datum
    adjusted_bathymetry_downscaled = downscale_raster(adjusted_bathymetry)
    
    initial_depths            = ( (0.97 - adjusted_bathymetry)            > 0) * (0.97 - adjusted_bathymetry)
    initial_depths_downscaled = ( (0.97 - adjusted_bathymetry_downscaled) > 0) * (0.97 - adjusted_bathymetry_downscaled)
    
    nrows, ncols = bathymetry.shape
    
    # obtained by checking difference between items in x array after running input-data/plot_bathy.m
    cellsize = 0.01
    
    # numerical values obtained by checking the first item of the x and y arrays after running input-data/plot_bathy.m
    xmin =   0.0120010375976563 - cellsize / 2
    ymin = -13.2609996795654    - cellsize / 2
    
    write_raster_file(
        raster=adjusted_bathymetry,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols,
        filename="oregon-seaside-0p01m.dem"
    )
    
    write_raster_file(
        raster=initial_depths,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols,
        filename="oregon-seaside-0p01m.start"
    )
    
    # at 0.02 m resolution instead of 0.01 m
    write_raster_file(
        raster=adjusted_bathymetry_downscaled,
        xmin=xmin,
        ymin=ymin,
        cellsize=2*cellsize,
        nrows=int(nrows/2),
        ncols=int(ncols/2),
        filename="oregon-seaside-0p02m.dem"
    )
    
    write_raster_file(
        raster=initial_depths_downscaled,
        xmin=xmin,
        ymin=ymin,
        cellsize=2*cellsize,
        nrows=int(nrows/2),
        ncols=int(ncols/2),
        filename="oregon-seaside-0p02m.start"
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
    
    write_stage_file()

def write_parameter_file(
    epsilon,
    solver,
    filename
):
    params = (
        "test_case     0\n" +
        "max_ref_lvl   12\n" +
        "min_dt        1\n" +
        "respath       results\n" +
        "epsilon       %s\n" +
        "fpfric        0.01\n" +
        "rasterroot    oregon-seaside-0p02m\n" +
        "bcifile       oregon-seaside.bci\n" +
        "bdyfile       oregon-seaside.bdy\n" +
        "stagefile     oregon-seaside.stage\n" +
        "tol_h         1e-3\n" +
        "tol_q         0\n" +
        "tol_s         1e-9\n" +
        "limitslopes   off\n" +
        "tol_Krivo     10\n" +
        "g             9.80665\n" +
        "saveint       4\n" +
        "massint       0.4\n" +
        "sim_time      40\n" +
        "solver        %s\n" +
        "cumulative    on\n" +
        "vtk           off\n" +
        "raster_out    on\n" +
        "voutput_stage on\n" +
        "wall_height   2.5"
    ) % (
        epsilon,
        solver
    )
    
    with open(filename, 'w') as fp:
        fp.write(params)

def run_simulation():
    if len(sys.argv) < 4: EXIT_HELP()
    
    dummy, option, solver, epsilon = sys.argv
    
    parameter_filename = "oregon-seaside.par"
    
    write_parameter_file(
        epsilon=epsilon,
        solver=solver,
        filename=parameter_filename
    )
    
    executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
    
    subprocess.run( [os.path.join("..", executable), parameter_filename] )
    
def main():
    if len(sys.argv) < 2: EXIT_HELP()
    
    option = sys.argv[1]
    
    if   option == "preprocess":
        write_all_input_files()
    elif option == "simulate":
        run_simulation()
    else:
        EXIT_HELP()

if __name__ == "__main__":
    main()