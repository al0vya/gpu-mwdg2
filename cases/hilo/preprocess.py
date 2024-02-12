import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "Use this tool as: python preprocess.py <SOLVER>, SOLVER={hwfv1|mwdg2} to select either the GPU-HWFV1 or GPU-MWDG2 solver, respectively."
    )
    
    sys.exit(help_message)

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
    
    fig.savefig(filename + ".png", bbox_inches="tight")

def load_grid_data():
    # 1st col is x coords in arcsecs
    # 2nd col is y coords in arcsecs
    # 3rd col is bathymetry elevation in metres
    return np.loadtxt( fname=os.path.join("input-data", "hilo_grid_1_3_arcsec.txt") )

def write_raster_files(
    grid_data,
    nrows,
    ncols
):
    print("Preparing raster files...")
    
    xmin     = 0
    ymin     = 0
    cellsize = 10
    
    dem = grid_data[:,2].reshape( (nrows,ncols) )
    
    # multiply by -1 to flip elevation
    dem *= -1
    
    # maximum depth of -30 m
    dem = np.maximum( dem, np.full(shape=(nrows,ncols), fill_value=-30) )
    
    # adjust for -30 m datum to make all elevations non-negative
    dem -= -30
    
    # initial water surface elevation of 30 m: start of tsunami time series
    depths = np.maximum( 30 - dem, np.zeros( shape=(nrows,ncols) ) )
    
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
        fname="hilo.dem",
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
        filename="hilo.dem"
    )
    
    np.savetxt(
        fname="hilo.start",
        X=np.flipud(depths),
        fmt="%.8f",
        header=header,
        comments=""
    )
    
    check_raster_file(
        raster=depths,
        xmin=xmin,
        ymin=ymin,
        cellsize=cellsize,
        nrows=nrows,
        ncols=ncols,
        filename="hilo.start"
    )
    
def write_bci_file():
    print("Preparing boundary condition file...")
    
    with open("hilo.bci", 'w') as fp:
        fp.write("N 0 7010 HVAR INFLOW")
        
def write_bdy_file():
    print("Preparing boundary time series file...")
    
    inflow_timeseries = np.loadtxt( fname=os.path.join("input-data", "se.txt") )
    
    # tsunami signal only
    inflow_timeseries[:,1] -= inflow_timeseries[0,1]
    
    # adjust by -30 m datum from DEM
    inflow_timeseries[:,1] -= -30
    
    with open("hilo.bdy", 'w') as fp:
        fp.write("\nINFLOW\n")
        
        timeseries_len = inflow_timeseries.shape[0]
        
        fp.write(str(timeseries_len) + " minutes\n")
        
        timeshift = inflow_timeseries[0,0]
        
        for entry in inflow_timeseries:
            fp.write(str( entry[1] ) + " " + str(entry[0] - timeshift) + "\n")
    
def plot_impact_wave():
    inflow_timeseries = np.loadtxt( fname=os.path.join("input-data", "se.txt") )
    
    # tsunami signal only
    inflow_timeseries[:,1] -= inflow_timeseries[0,1]
    
    fig, ax = plt.subplots( figsize=(4,1.5) )
    
    ax.plot(inflow_timeseries[:,0]/60, inflow_timeseries[:,1])
    
    ax.set_title("Impact wave at top boundary")
    ax.set_xlabel("$t$ (hr)")
    ax.set_ylabel("$h + z$ (m)")
    
    #ax.set_xlim( (8.5,11) )
    ax.set_ylim( (-1.1,1.1) )
    
    fig.tight_layout()
    
    fig.savefig("impact-wave.png", bbox_inches="tight")

def convert_from_arcsecs_to_metres(
    xmin,
    ymin,
    x,
    y,
    cellsize_arcsecs,
    cellsize_metres=10
):
    # find number of cells using arcsec measurements
    x_cells = (x - xmin) / cellsize_arcsecs
    y_cells = (y - ymin) / cellsize_arcsecs
    
    # then convert from number of cells to metres
    x_metres = x_cells * cellsize_metres
    y_metres = y_cells * cellsize_metres
    
    return (x_metres, y_metres)

def write_stage_file(
    grid_data,
    nrows,
    ncols
):
    print("Preparing stage file...")
    
    x = grid_data[:,0].reshape( (nrows, ncols) )
    
    # average difference between x coords to compute cellsize
    cellsize = np.average( x[0,1:] - x[0,:-1] )
    
    # from hilo_grid_1_3_arcsec.txt
    xmin = 204.90028
    ymin = 19.71000
    
    # from http://coastal.usc.edu/currents_workshop/problems/prob2.html
    x_control_point = 204.93
    y_control_point = 19.7576
    
    x_tide_station = 204.9447
    y_tide_station = 19.7308
    
    x_adcp_1125 = 204.9180
    y_adcp_1125 = 19.7452
    
    x_adcp_1126 = 204.9300
    y_adcp_1126 = 19.7417
    
    x_control_point, y_control_point = convert_from_arcsecs_to_metres(
        xmin=xmin,
        ymin=ymin,
        x=x_control_point,
        y=y_control_point,
        cellsize_arcsecs=cellsize
    )
    
    x_tide_station, y_tide_station = convert_from_arcsecs_to_metres(
        xmin=xmin,
        ymin=ymin,
        x=x_tide_station,
        y=y_tide_station,
        cellsize_arcsecs=cellsize
    )
    
    x_adcp_1125, y_adcp_1125 = convert_from_arcsecs_to_metres(
        xmin=xmin,
        ymin=ymin,
        x=x_adcp_1125,
        y=y_adcp_1125,
        cellsize_arcsecs=cellsize
    )
    
    x_adcp_1126, y_adcp_1126 = convert_from_arcsecs_to_metres(
        xmin=xmin,
        ymin=ymin,
        x=x_adcp_1126,
        y=y_adcp_1126,
        cellsize_arcsecs=cellsize
    )
    
    with open("hilo.stage", 'w') as fp:
        stages = (
            "5\n" +
            "3000 6910\n" +
            "%s %s\n" +
            "%s %s\n" +
            "%s %s\n" +
            "%s %s"
        ) % (
            x_control_point, y_control_point,
            x_tide_station,  y_tide_station,
            x_adcp_1125,     y_adcp_1125,
            x_adcp_1126,     y_adcp_1126
        )
        
        fp.write(stages)

def write_par_file(solver):
    print("Preparing parameter file...")
    
    params = (
        "cuda\n" +
        f"{solver}\n" +
        "cumulative\n" +
        "raster_out\n" +
        "voutput_stage\n" +
        "refine_wall\n" +
        "ref_thickness 16\n" +
        "massint       10\n" +
        "saveint       23000\n" +
        "DEMfile       hilo.dem\n" +
        "startfile     hilo.start\n" +
        "bcifile       hilo.bci\n" +
        "bdyfile       hilo.bdy\n" +
        "stagefile     hilo.stage\n" +
        "sim_time      23000\n" +
        "fpfric        0.025\n" +
        "max_ref_lvl   10\n" +
        "epsilon       0\n" +
        "wall_height   180\n" +
        "initial_tstep 1\n"
    )
    
    with open("hilo.par", 'w') as fp:
        fp.write(params)

def write_all_input_files(solver):
    write_bci_file()
    
    write_bdy_file()

    nrows = 692
    ncols = 701
    
    grid_data = load_grid_data()
    
    write_raster_files(
        grid_data=grid_data,
        nrows=nrows,
        ncols=ncols
    )
    
    write_stage_file(
        grid_data=grid_data,
        nrows=nrows,
        ncols=ncols
    )
    
    plot_impact_wave()
    
    write_par_file(solver)
    
def main():
    if len(sys.argv) < 2:
        EXIT_HELP()
        
    solver = sys.argv[1]
    
    if solver != "hwfv1" and solver != "mwdg2":
        EXIT_HELP()
        
    write_all_input_files(solver)

if __name__ == "__main__":
    main()