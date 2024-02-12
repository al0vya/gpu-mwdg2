import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "Use this tool as: python preprocess.py <SOLVER>, SOLVER={hwfv1|mwdg2} to select either the GPU-HWFV1 or GPU-MWDG2 solver, respectively."
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
    
    fig, ax = plt.subplots( figsize=(5,1.5) )
    
    wave_amplitude_original     = 0.51
    wave_amplitude_calibrated   = 0.60
    scale_factor_wave_amplitude = wave_amplitude_calibrated / wave_amplitude_original
    
    bdy_inlet = scale_factor_wave_amplitude * (0.97 + boundary_timeseries[:,1] - datum)
    
    # timeshift to account for wave travel time from x = 0 m to 5 m    
    timeshift = 16 - 14.75
    
    ax.plot(
        boundary_timeseries[:,0]-timeshift,
        (bdy_inlet-bdy_inlet[0])/scale_factor_wave_amplitude
    )
    
    plt.setp(
        ax,
        title="Impact wave",
        xlim=( (0,40) ),
        xlabel=r"$t \, (s)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    fig.savefig("impact-wave.svg", bbox_inches="tight")
    
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

def plot_impact_wave_direction(ax):
    xmax = (7010    ) / 1000
    ymax = (6920-100) / 1000
    
    num_arrows = 5
    
    gap_between_arrows = xmax / num_arrows
    
    arrow_centres = [0.5 * gap_between_arrows + i * gap_between_arrows for i in range(num_arrows)]
    
    for centre in arrow_centres:
        ax.arrow(x=centre, y=ymax, dx=0, dy=-2500 / 1000, head_width=200 / 1000, color='r')
        
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
    
    y = [ (ymin + j * cellsize) / 1000 for j in range(nrows) ]
    x = [ (xmin + i * cellsize) / 1000 for i in range(ncols) ]
    
    x, y = np.meshgrid(x, y)
    
    contourset = ax.contourf(x, y, raster, levels=30)
    
    fig.colorbar(
        contourset,
        orientation="horizontal",
        label='m'
    )
    
    control_point = (3209.998457028253 / 1000,  5141.1819163712990 / 1000)
    tide_gauge    = (4797.716401790525 / 1000,  2246.5668878259044 / 1000)
    adcp_HA1125   = (1913.902175590039 / 1000,  3801.8824255519144 / 1000)
    adcp_HA1126   = (3209.998457028253 / 1000,  3423.8543434658964 / 1000)
        
    ax.scatter(control_point[0], control_point[1], marker='x', label="Control point")
    ax.scatter(tide_gauge[0],    tide_gauge[1],    marker='x', label="Tide gauge")
    ax.scatter(adcp_HA1125[0],   adcp_HA1125[1],   marker='x', label="ADCP HA1125")
    ax.scatter(adcp_HA1126[0],   adcp_HA1126[1],   marker='x', label="ADCP HA1126")
    
    plot_impact_wave_direction(ax)
    
    ax.legend(loc="lower left", fontsize=8)
    
    plt.setp(
        ax,
        xlabel="$x$ (km)",
        ylabel="$y$ (km)"
    )
    
    fig.tight_layout()
    
    fig.savefig(filename + ".svg", bbox_inches="tight")
    
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

def write_all_input_files(solver):
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
    
    write_par_file(solver)

def write_par_file(solver):
    params = (
        f"{solver}\n" +
        "cuda\n" +
        "cumulative\n" +
        "raster_out\n" +
        "max_ref_lvl   12\n" +
        "initial_tstep 1\n" +
        "epsilon       0\n" +
        "fpfric        0.025\n" + # from "A comparison of a two-dimensional depth-averaged flow model ... for predicting tsunami ..."
        "DEMfile       oregon-seaside-0p02m.dem\n" +
        "startfile     oregon-seaside-0p02m.start\n" +
        "stagefile     oregon-seaside.stage\n" +
        "refine_wall   on\n" +
        "ref_thickness 16\n" +
        "massint       0.2\n" +
        "saveint       39.7\n" +
        "sim_time      39.7\n" +
        "wall_height   2.5\n" +
        "voutput_stage\n"
    )
    
    with open("oregon-seaside.par", 'w') as fp:
        fp.write(params)

def main():
    if len(sys.argv) < 2:
        EXIT_HELP()
    
    solver = sys.argv[1]
    
    if solver != "hwfv1" and solver != "mwdg2":
        EXIT_HELP()
        
    write_all_input_files(solver)

if __name__ == "__main__":
    main()