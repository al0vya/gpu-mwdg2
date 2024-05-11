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

def load_experimental_gauge_timeseries():
    print("Loading experimental timeseries...")
    
    port_data  = np.loadtxt( fname=os.path.join("input-data", "port_data.txt" ) )
    tide_gauge = np.loadtxt( fname=os.path.join("input-data", "tide_gauge.txt") )
    
    A_Beacon = {
        "time"    : port_data[:,0+0],
        "tsunami" : port_data[:,1+0],
        "total"   : port_data[:,2+0],
        "tidal"   : port_data[:,2+0] - port_data[:,1+0],
    }
    
    Tug_Harbour = {
        "time"    : port_data[:,0+3],
        "tsunami" : port_data[:,1+3],
        "total"   : port_data[:,2+3],
        "tidal"   : port_data[:,2+3] - port_data[:,1+3],
    }
    
    Sulphur_Point = {
        "time"    : port_data[:,0+6],
        "tsunami" : port_data[:,1+6],
        "total"   : port_data[:,2+6],
        "tidal"   : port_data[:,2+6] - port_data[:,1+6],
    }
    
    Taut = {
        "time"    : tide_gauge[:,0+0],
        "tsunami" : tide_gauge[:,1+0],
        "total"   : tide_gauge[:,2+0],
        "tidal"   : tide_gauge[:,2+0] - tide_gauge[:,1+0],
    }
    
    Moturiki = {
        "time"    : tide_gauge[:,0+3],
        "tsunami" : tide_gauge[:,1+3],
        "total"   : tide_gauge[:,2+3],
        "tidal"   : tide_gauge[:,2+3] - tide_gauge[:,1+3],
    }
    
    return {
        "A_Beacon"      : A_Beacon,
        "Tug_Harbour"   : Tug_Harbour,
        "Sulphur_Point" : Sulphur_Point,
        "Taut"          : Taut,
        "Moturiki"      : Moturiki
    }

def find_min_arrays(*arrays):
    a = np.amin( arrays[0] )
    
    for array in arrays:
        a = np.min( [ a, np.amin(array) ] )
    
    return a

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

def write_raster_file(
    raster,
    nrows,
    ncols,
    cellsize,
    filename
):
    print(f"Preparing raster file: {filename}")
    
    header = (
        "ncols        %s\n" +
        "nrows        %s\n" +
        "xllcorner    0\n" +
        "yllcorner    0\n" +
        "cellsize     %s\n" +
        "NODATA_value -9999"
    ) % (
        ncols,
        nrows,
        cellsize
    )
    
    np.savetxt(
        fname=filename,
        X=np.flipud(raster),
        fmt="%.8f",
        header=header,
        comments=""
    )

def plot_impact_wave_direction(ax):
    # convert from m to km
    xmax = 4096 * 10 / 1000
    ymax = 2241 * 10 / 1000 - 0.1
    
    num_arrows = 5
    
    gap_between_arrows = xmax / num_arrows
    
    arrow_centres = [0.5 * gap_between_arrows + i * gap_between_arrows for i in range(num_arrows)]
    
    for centre in arrow_centres:
        ax.arrow(x=centre, y=ymax, dx=0, dy=-5, head_width=0.66, color='r')

def check_raster_file(
    raster,
    nrows,
    ncols,
    xmin,
    ymin,
    cellsize,
    filename
):
    x = [ xmin + cellsize * i for i in range(ncols) ]
    y = [ ymin + cellsize * j for j in range(nrows) ]
    
    x, y = np.meshgrid(x, y)
    
    z_max    = np.max(raster)
    z_min    = 0 # to avoid -9999 NODATA_value otherwise obtained using np.min(raster)
    levels   = 30
    dz       = (z_max - z_min) / levels
    z_levels = [ z_min + dz * i for i in range(levels+1) ]
    
    fig, ax    = plt.subplots( figsize=(5.0,4.2) )
    
    contourset = ax.contourf(
        x / 1000, # convert from m to km
        y / 1000,
        raster,
        levels=z_levels
    )
    
    colorbar = fig.colorbar(
        contourset,
        orientation="horizontal",
        label=r"$m$"
    )
    
    stages = [
        (2.724e4, 1.846e4, "A Beacon"),
        (3.085e4, 1.512e4, "Tug Harbour"),
        (3.200e4, 1.347e4, "Sulphur Point"),
        (3.005e4, 1.610e4, "Moturiki"),
        (2.925e4, 1.466e4, "ADCP")
    ]
    
    for stage in stages:
        ax.scatter(
            stage[0] / 1000, # convert from m to km
            stage[1] / 1000,
            linewidth=0.75,
            marker='x',
            label=stage[2]
        )
    
    plot_impact_wave_direction(ax)
    
    ax.legend()
    ax.set_xlabel("$x$ (km)")
    ax.set_ylabel("$y$ (km)")
    
    fig.savefig(filename + ".png", bbox_inches="tight")
    
def write_bci_file(
    ncols,
    cellsize,
    timeseries_name
):
    print("Preparing bci file...")
    
    xmax = ncols * cellsize
    
    with open("tauranga.bci", 'w') as fp:
        fp.write( "N 0 %s HVAR %s" % (xmax, timeseries_name) )
        
def write_bdy_file(
    time,
    series,
    timeseries_name
):
    print("Preparing bdy file...")
    
    timeseries_len = len(series)
    
    with open("tauranga.bdy", 'w') as fp:
        header = (
            "%s\n" +
            "%s hours\n"
        ) % (
            timeseries_name,
            timeseries_len,
        )
        
        fp.write(header)
        
        for entry in zip(time, series):
            fp.write(str( entry[1] ) + " " + str( entry[0] ) + "\n")
        
def write_stage_file():
    print("Preparing stage file...")
    
    with open("tauranga.stage", 'w') as fp:
        stages = (
            "5\n"
            "2.724e4 1.846e4\n" + # A Beacon
            "3.085e4 1.512e4\n" + # Tug Harbour
            "3.200e4 1.347e4\n" + # Sulphur Point
            "3.005e4 1.610e4\n" + # Moturiki
            "2.925e4 1.466e4\n"   # ADCP
        )
        
        fp.write(stages)

def plot_impact_wave(
    time,
    series
):
    fig, ax = plt.subplots( figsize=(5,1) )
    
    ax.plot(
        time,
        series
    )
    
    plt.setp(
        ax,
        title="Impact wave at top boundary",
        xlim=( (0,40) ),
        ylim=( (-1,1) ),
        xlabel="$t$ (hr)",
        ylabel="$h + z$ (m)"
    )
    
    fig.savefig("impact-wave.png", bbox_inches="tight")

def write_all_input_files(solver):
    dem = np.loadtxt(fname=os.path.join("input-data", "bathymetry.csv"), delimiter=",")
    
    NODATA_mask = (dem + 9999 <= 1e-10) == 0
    
    # find datum whilst ignoring NODATA_values
    datum = np.min(dem * NODATA_mask)
    
    # adjusting DEM by datum; avoiding subtracting from pixels with NODATA_value
    dem -= NODATA_mask * datum
    
    depths = np.maximum( 36.973091089 - dem, np.zeros(dem.shape) ) * NODATA_mask
    
    # same number of rows as csv file
    nrows = dem.shape[0]
    
    # but 4096 cols to allow L = 12
    ncols = 4096
    
    cellsize = 10
    
    write_raster_file(
        raster=dem[:,:ncols], # adjust datum only for non-NODATA_values
        nrows=nrows,
        ncols=ncols,
        cellsize=cellsize,
        filename="tauranga.dem"
    )
    
    check_raster_file(
        raster=dem[:,:ncols],
        nrows=nrows,
        ncols=ncols,
        xmin=0,
        ymin=0,
        cellsize=cellsize,
        filename="tauranga.dem"
    )
    
    write_raster_file(
        raster=depths[:,:ncols],
        nrows=nrows,
        ncols=ncols,
        cellsize=cellsize,
        filename="tauranga.start"
    )
    
    check_raster_file(
        raster=depths[:,:ncols],
        nrows=nrows,
        ncols=ncols,
        xmin=0,
        ymin=0,
        cellsize=cellsize,
        filename="tauranga.start"
    )
    
    timeseries_name = "INLET"
    
    write_bci_file(
        ncols=ncols,
        cellsize=cellsize,
        timeseries_name=timeseries_name
    )
    
    gauges = load_experimental_gauge_timeseries()
    
    write_bdy_file(
        time=gauges["A_Beacon"]["time"],
        series=( gauges["A_Beacon"]["total"] - datum ),
        timeseries_name=timeseries_name
    )
    
    plot_impact_wave(
        time=gauges["A_Beacon"]["time"],
        series=gauges["A_Beacon"]["total"]
    )
    
    write_stage_file()
    
    write_par_file(solver)

def write_par_file(solver):
    params = (
        f"{solver}\n" +
        "cuda\n" +
        "max_ref_lvl   12\n" +
        "initial_tstep 1\n" +
        "epsilon       0\n" +
        "fpfric        0.025\n" +
        "tol_h         1e-2\n" +
        "DEMfile       tauranga.dem\n" +
        "startfile     tauranga.start\n" +
        "bcifile       tauranga.bci\n" +
        "bdyfile       tauranga.bdy\n" +
        "stagefile     tauranga.stage\n" +
        "massint       500\n" +
        "saveint       144000\n" +
        "sim_time      144000\n" +
        "raster_out\n" +
        "cumulative\n" +
        "voutput_stage\n" +
        "wall_height   420"
    )
    
    with open("tauranga.par", 'w') as fp:
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