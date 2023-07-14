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
    
    wave_amplitude_original     = 0.51
    wave_amplitude_calibrated   = 0.60
    scale_factor_wave_amplitude = wave_amplitude_calibrated / wave_amplitude_original
    
    bdy_inlet = scale_factor_wave_amplitude * (0.97 + boundary_timeseries[:,1] - datum)
    
    ax.plot(
        boundary_timeseries[:,0],
        bdy_inlet
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
        
        for time, inlet in zip(boundary_timeseries[:,0], bdy_inlet):
            fp.write(str(inlet) + " " + str(time) + "\n")

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
    
    contourset = ax.contourf(x, y, raster, levels=30)
    
    fig.colorbar(
        contourset,
        orientation="horizontal",
        label=r"$m$"
    )
    
    gauges_A = [
        (33.611, -3.193),
        (34.103, -3.194),
        (34.534, -3.184),
        (35.040, -3.181),
        (35.544, -3.194),
        (36.355, -3.199),
        (37.767, -3.201),
        (39.223, -3.204),
        (40.767, -3.228)
    ]
    
    gauges_B = [
        (33.721, -0.588),
        (34.218, -0.533),
        (34.679, -0.467),
        (35.176, -0.406),
        (35.747, -0.317),
        (36.635, -0.229),
        (37.773, -0.068),
        (39.218,  0.135),
        (40.668,  0.269)
    ]
    
    gauges_C = [
        (33.809, 1.505),
        (34.553, 1.604),
        (35.051, 1.686),
        (35.556, 1.769),
        (36.050, 1.845),
        (37.047, 1.988),
        (38.243, 2.193),
        (39.208, 2.338),
        (40.400, 2.582)
    ]
    
    gauges_D = [
        (35.124, 3.712),
        (36.684, 3.888),
        (39.086, 4.070),
        (38.141, 3.585)
    ]
    
    gauges_W = [
        ( 2.068,-0.515),
        ( 2.068, 4.605),
        (18.618, 0.000),
        (18.618, 2.860)
    ]
    
    plt.scatter(
        [gauge[0] for gauge in gauges_A],
        [gauge[1] for gauge in gauges_A],
        marker='x',
        linewidth=0.5,
        facecolor='r',
        s=10,
        label="A"
    )
    
    plt.scatter(
        [gauge[0] for gauge in gauges_B],
        [gauge[1] for gauge in gauges_B],
        marker='x',
        linewidth=0.5,
        facecolor='k',
        s=10,
        label="B"
    )
    
    plt.scatter(
        [gauge[0] for gauge in gauges_C],
        [gauge[1] for gauge in gauges_C],
        marker='x',
        linewidth=0.5,
        facecolor='m',
        s=10,
        label="C"
    )
    
    plt.scatter(
        [gauge[0] for gauge in gauges_D],
        [gauge[1] for gauge in gauges_D],
        marker='x',
        linewidth=0.5,
        facecolor='b',
        s=10,
        label="D"
    )
    
    ax.legend()
    
    plt.setp(
        ax,
        xlabel=r"$x \, (m)$",
        ylabel=r"$y \, (m)$"
    )
    
    fig.tight_layout()
    
    fig.savefig(os.path.join("results", filename + ".svg"), bbox_inches="tight")
    
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
        "36\n" +
        " 5.000   0.000\n" + # boundary cell
        " 2.068  -0.515\n" + # W1
        " 2.068   4.605\n" + # W2
        "18.618   0.000\n" + # W3
        "18.618   2.860\n" + # W4
        "33.611  -3.193\n" + # A1
        "34.103  -3.194\n" + # A2
        "34.534  -3.184\n" + # A3
        "35.040  -3.181\n" + # A4
        "35.544  -3.194\n" + # A5
        "36.355  -3.199\n" + # A6
        "37.767  -3.201\n" + # A7
        "39.223  -3.204\n" + # A8
        "40.767  -3.228\n" + # A9
        "33.721  -0.588\n" + # B1
        "34.218  -0.533\n" + # B2
        "34.679  -0.467\n" + # B3
        "35.176  -0.406\n" + # B4
        "35.747  -0.317\n" + # B5
        "36.635  -0.229\n" + # B6
        "37.773  -0.068\n" + # B7
        "39.218   0.135\n" + # B8
        "40.668   0.269\n" + # B9
        "33.809   1.505\n" + # C1
        "34.553   1.604\n" + # C2
        "35.051   1.686\n" + # C3
        "35.556   1.769\n" + # C4
        "36.050   1.845\n" + # C5
        "37.047   1.988\n" + # C6
        "38.243   2.193\n" + # C7
        "39.208   2.338\n" + # C8
        "40.400   2.582\n" + # C9
        "35.124   3.712\n" + # D1
        "36.684   3.888\n" + # D2
        "39.086   4.070\n" + # D3
        "38.141   3.585\n"   # D4
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
    
    '''
    # remove western cells because inlet wave forced at x = 5 m
    # if using water surface elevation time series
    western_cells_to_trim = int( ( 5 - (xmin) ) / cellsize )
    
    write_raster_file(
        raster=adjusted_bathymetry[:,western_cells_to_trim:],
        xmin=5,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols-western_cells_to_trim,
        filename="oregon-seaside-0p01m.dem"
    )
    
    write_raster_file(
        raster=initial_depths[:,western_cells_to_trim:],
        xmin=5,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols-western_cells_to_trim,
        filename="oregon-seaside-0p01m.start"
    )
    '''
    
    # if using velocity time series
    write_raster_file(
        raster=downscale_raster(adjusted_bathymetry),
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize*2,
        nrows=int(nrows/2),
        ncols=int(ncols/2),
        filename="oregon-seaside-0p02m.dem"
    )
    
    write_raster_file(
        raster=downscale_raster(initial_depths),
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize*2,
        nrows=int(nrows/2),
        ncols=int(ncols/2),
        filename="oregon-seaside-0p02m.start"
    )
    
    timeseries_name = "INLET"
    
    # if using water surface elevation time series
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
    results_dir,
    filename
):
    params = (
        "test_case     0\n" +
        "max_ref_lvl   12\n" +
        "initial_tstep        1\n" +
        "respath       %s\n" +
        "epsilon       %s\n" +
        "fpfric        0.025\n" + # from "A comparison of a two-dimensional depth-averaged flow model ... for predicting tsunami ..."
        "rasterroot    oregon-seaside-0p02m\n" +
        "bcifile       oregon-seaside.bci\n" +
        "bdyfile       oregon-seaside.bdy\n" +
        "stagefile     oregon-seaside.stage\n" +
        "tol_h         1e-3\n" +
        "tol_q         0\n" +
        "tol_s         1e-9\n" +
        "limitslopes   off\n" +
        "tol_Krivo     10\n" +
        "refine_wall   on\n" +
        "ref_thickness 16\n" +
        "g             9.80665\n" +
        "massint       0.2\n" +
        "saveint       39.7\n" +
        "sim_time      39.7\n" +
        "solver        %s\n" +
        "vtk           on\n" +
        "cumulative    on\n" +
        "voutput_stage on\n" +
        "wall_height   2.5"
    ) % (
        results_dir,
        epsilon,
        solver
    )
    
    with open(filename, 'w') as fp:
        fp.write(params)

def run_simulation():
    if len(sys.argv) != 5: EXIT_HELP()
    
    dummy, option, solver, epsilon, results_dir = sys.argv
    
    parameter_filename = "oregon-seaside.par"
    
    write_parameter_file(
        epsilon=epsilon,
        solver=solver,
        results_dir=results_dir,
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