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
        0.97 + boundary_timeseries[:,1] - datum
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
            fp.write(str(0.97 + entry[1] - datum) + " " + str( entry[0] ) + "\n")

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
        "6\n" +
        " 5.000  0.000\n" + # boundary cell
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
    
    print( "Datum: " + str(datum) )
    
    adjusted_bathymetry = bathymetry - datum
    
    initial_depths = ( (0.97 - adjusted_bathymetry) > 0 ) * (0.97 - adjusted_bathymetry)
    
    nrows, ncols = bathymetry.shape
    
    # obtained by checking difference between items in x array after running input-data/plot_bathy.m
    cellsize = 0.01
    
    # numerical values obtained by checking the first item of the x and y arrays after running input-data/plot_bathy.m
    xmin =   0.012 - cellsize / 2
    ymin = -13.260 - cellsize / 2
    
    # remove western cells because inlet wave forced at x = 5 m
    western_cells_to_trim = int( ( 5 - (xmin) ) / cellsize )
    
    xmin = 5
    
    write_raster_file(
        raster=adjusted_bathymetry[:,western_cells_to_trim:],
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols-western_cells_to_trim,
        filename="oregon-seaside-0p01m.dem"
    )
    
    write_raster_file(
        raster=initial_depths[:,western_cells_to_trim:],
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols-western_cells_to_trim,
        filename="oregon-seaside-0p01m.start"
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
        "fpfric        0.025\n" + # from "A comparison of a two-dimensional depth-averaged flow model ... for predicting tsunami ..."
        "rasterroot    oregon-seaside-0p01m\n" +
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

def load_experimental_gauge_timeseries():
    experimental_gauges = {}
    
    experimental_gauges["W3"] = {}
    experimental_gauges["B1"] = {}
    experimental_gauges["B4"] = {}
    experimental_gauges["B6"] = {}
    experimental_gauges["B9"] = {}
    
    wavegage = np.loadtxt(fname="Wavegage.txt", skiprows=1)
    
    experimental_gauges["W3"]["time"]    = wavegage[:,0]
    experimental_gauges["W3"]["history"] = wavegage[:,5]
    
    return experimental_gauges

def load_computed_gauge_timeseries(
    stagefile
):
    print("Loading computed gauges timeseries: %s..." % stagefile)
    
    gauges = np.loadtxt(os.path.join("results", stagefile), skiprows=11, delimiter=" ")
    
    datum = -0.00202286243437291
    
    return {
        "time" : gauges[:,0],
        "W3"   : gauges[:,1] + datum,
        "B1"   : gauges[:,2] + datum,
        "B4"   : gauges[:,3] + datum,
        "B6"   : gauges[:,4] + datum,
        "B9"   : gauges[:,5] + datum
    }
    
def read_stage_elevations(
    stagefile
):
    header = []
    
    with open(os.path.join("results", stagefile), 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 7:
                break
    
    return {
        "W3" : float( header[0].split()[3] ),
        "B1" : float( header[1].split()[3] ),
        "B4" : float( header[2].split()[3] ),
        "B6" : float( header[3].split()[3] ),
        "B9" : float( header[4].split()[3] )
    }

def compare_timeseries_stage(
    stagefiles,
    experimental_gauges,
    name
):
    my_rc_params = {
        "legend.fontsize" : "large",
        "axes.labelsize"  : "large",
        "axes.titlesize"  : "large",
        "xtick.labelsize" : "large",
        "ytick.labelsize" : "large"
    }
    
    plt.rcParams.update(my_rc_params)
    
    fig, ax = plt.subplots()
    
    for stagefile in stagefiles:
        elevations      = read_stage_elevations(stagefile)
        computed_gauges = load_computed_gauge_timeseries(stagefile)
        
        ax.plot(
            computed_gauges["time"],
            computed_gauges[name] + elevations[name] - 0.97,
            label=stagefile
        )
    
    ax.plot(
        experimental_gauges[name]["time"],
        experimental_gauges[name]["history"],
        label="experimental"
    )
    
    ylim = (
        np.min( experimental_gauges[name]["history"] ),
        np.max( experimental_gauges[name]["history"] )
    )
    
    plt.setp(
        ax,
        title=name,
        xlim=( computed_gauges["time"][0], computed_gauges["time"][-1] ),
        ylim=ylim,
        xlabel=r"$t \, (hr)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    plt.legend()
    
    fig.savefig(name, bbox_inches="tight")
    
    plt.close()
    
def compare_timeseries_all_stages():
    stagefiles = [
        #"stage-hwfv1-1e-3.wd",
        "stage-mwdg2-1e-3.wd"
        ]
     
    experimental_gauges = load_experimental_gauge_timeseries()
    
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W3")
    #compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="")
    #compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="")
    #compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="")

def main():
    if len(sys.argv) < 2: EXIT_HELP()
    
    option = sys.argv[1]
    
    if   option == "preprocess":
        write_all_input_files()
    elif option == "simulate":
        run_simulation()
    elif option == "postprocess":
        compare_timeseries_all_stages()
    else:
        EXIT_HELP()

if __name__ == "__main__":
    main()