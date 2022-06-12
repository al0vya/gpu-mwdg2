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

def plot_tsunami_signal(
    A_Beacon,
    Tug_Harbour,
    Sulphur_Point,
    Moturiki
):
    fig, axs = plt.subplots(nrows=4)
    
    axs[0].plot( A_Beacon[0],      A_Beacon[1] )
    axs[1].plot( Tug_Harbour[0],   Tug_Harbour[1] )
    axs[2].plot( Sulphur_Point[0], Sulphur_Point[1] )
    axs[3].plot( Moturiki[0],      Moturiki[1] )
    
    plt.setp( axs, xlim=(0,40), ylim=(-0.5,0.5) )
    
    plt.setp(axs[0], title="A Beacon")
    plt.setp(axs[1], title="Tug Harbour")
    plt.setp(axs[2], title="Sulphur Point")
    plt.setp(axs[3], title="Moturiki")
    
    fig.tight_layout()
    
    plt.savefig(os.path.join("results", "tsunami-signal"), bbox_inches="tight")

def plot_total_signal(
    A_Beacon,
    Tug_Harbour,
    Sulphur_Point,
    Moturiki
):
    fig, axs = plt.subplots(nrows=4)
    
    axs[0].plot( A_Beacon[0],      A_Beacon[1] )
    axs[1].plot( Tug_Harbour[0],   Tug_Harbour[1] )
    axs[2].plot( Sulphur_Point[0], Sulphur_Point[1] )
    axs[3].plot( Moturiki[0],      Moturiki[1] )
    
    plt.setp( axs, xlim=(0,40), ylim=(-1.0,1.2) )
    
    plt.setp(axs[0], title="A Beacon")
    plt.setp(axs[1], title="Tug Harbour")
    plt.setp(axs[2], title="Sulphur Point")
    plt.setp(axs[3], title="Moturiki")
    
    fig.tight_layout()
    
    plt.savefig(os.path.join("results", "total-signal"), bbox_inches="tight")

def plot_tidal_signal(
    A_Beacon,
    Tug_Harbour,
    Sulphur_Point,
    Moturiki
):
    fig, axs = plt.subplots(nrows=4)
    
    axs[0].plot( A_Beacon[0],      A_Beacon[1] )
    axs[1].plot( Tug_Harbour[0],   Tug_Harbour[1] )
    axs[2].plot( Sulphur_Point[0], Sulphur_Point[1] )
    axs[3].plot( Moturiki[0],      Moturiki[1] )
    
    plt.setp( axs, xlim=(0,40), ylim=(-1.0,1.2) )
    
    plt.setp(axs[0], title="A Beacon")
    plt.setp(axs[1], title="Tug Harbour")
    plt.setp(axs[2], title="Sulphur Point")
    plt.setp(axs[3], title="Moturiki")
    
    fig.tight_layout()
    
    plt.savefig(os.path.join("results", "tide-signal"), bbox_inches="tight")
    
def plot_all_signals(gauges):
    t_A_Beacon      = gauges["A_Beacon"]["time"]
    t_Tug_Harbour   = gauges["Tug_Harbour"]["time"]
    t_Sulphur_Point = gauges["Sulphur_Point"]["time"]
    t_Moturiki      = gauges["Moturiki"]["time"]
    
    plot_tsunami_signal(
        A_Beacon      = ( t_A_Beacon,           gauges["A_Beacon"]["tsunami"] ),
        Tug_Harbour   = ( t_Tug_Harbour,     gauges["Tug_Harbour"]["tsunami"] ),
        Sulphur_Point = ( t_Sulphur_Point, gauges["Sulphur_Point"]["tsunami"] ),
        Moturiki      = ( t_Moturiki,           gauges["Moturiki"]["tsunami"] ),
    )
    
    plot_total_signal(
        A_Beacon      = ( t_A_Beacon,           gauges["A_Beacon"]["total"] ),
        Tug_Harbour   = ( t_Tug_Harbour,     gauges["Tug_Harbour"]["total"] ),
        Sulphur_Point = ( t_Sulphur_Point, gauges["Sulphur_Point"]["total"] ),
        Moturiki      = ( t_Moturiki,           gauges["Moturiki"]["total"] ),
    )
    
    plot_tidal_signal(
        A_Beacon      = ( t_A_Beacon,           gauges["A_Beacon"]["tidal"] ),
        Tug_Harbour   = ( t_Tug_Harbour,     gauges["Tug_Harbour"]["tidal"] ),
        Sulphur_Point = ( t_Sulphur_Point, gauges["Sulphur_Point"]["tidal"] ),
        Moturiki      = ( t_Moturiki,           gauges["Moturiki"]["tidal"] ),
    )

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

def write_bathymetry(
    bathymetry,
    nrows,
    ncols,
    cellsize,
    filename
):
    print("Preparing DEM file...")
    
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
        X=np.flipud(bathymetry),
        fmt="%.8f",
        header=header,
        comments=""
    )

def plot_bathymetry(
        bathymetry,
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
        
        z_max    = np.max(bathymetry)
        levels   = 30
        dz       = z_max / levels
        z_levels = [ 0 + dz * i for i in range(levels+1) ]
        
        fig, ax    = plt.subplots( figsize=(5.0,4.2) )
        
        contourset = ax.contourf(
            x / 1000, # convert from m to km
            y / 1000,
            bathymetry,
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
        
        ax.legend()
        ax.set_xlabel(r"$x \, (km)$")
        ax.set_ylabel(r"$y \, (km)$")
        
        fig.savefig(fname=(os.path.join("results", filename)), bbox_inches="tight")
        
        plt.close()
    
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
        
def find_datum():
    print("Loading bathymetry file...")
    
    bathymetry = np.loadtxt(fname=os.path.join("input-data", "bathymetry.csv"), delimiter=",")
    
    NODATA_mask = (bathymetry + 9999 <= 1e-10) == 0
    
    gauges = load_experimental_gauge_timeseries()
    
    print("Finding datum...")
    
    return find_min_arrays(
        bathymetry * NODATA_mask, # remove -9999 when finding datum
        gauges["A_Beacon"]["tsunami"],
        gauges["A_Beacon"]["total"],
        gauges["A_Beacon"]["tidal"],
        gauges["Tug_Harbour"]["tsunami"],
        gauges["Tug_Harbour"]["total"],
        gauges["Tug_Harbour"]["tidal"],
        gauges["Sulphur_Point"]["tsunami"],
        gauges["Sulphur_Point"]["total"],
        gauges["Sulphur_Point"]["tidal"],
        gauges["Moturiki"]["tsunami"],
        gauges["Moturiki"]["total"],
        gauges["Moturiki"]["tidal"],
    )

def write_all_input_files():
    datum = find_datum()
    
    print(datum)
    
    print("Adjusting for datum in bathymetry data...")
    
    bathymetry = np.loadtxt(fname=os.path.join("input-data", "bathymetry.csv"), delimiter=",")
    
    NODATA_mask = (bathymetry + 9999 <= 1e-10) == 0
    
    # same number of rows as csv file
    nrows = bathymetry.shape[0]
    
    # but 4096 cols to allow L = 12
    ncols = 4096
    
    cellsize = 10
    
    write_bathymetry(
        bathymetry=( bathymetry - (NODATA_mask * datum) )[:,:ncols], # adjust datum only for non-NODATA_values
        nrows=nrows,
        ncols=ncols,
        cellsize=cellsize,
        filename="tauranga-10m.dem"
    )
    
    write_bathymetry(
        bathymetry=downscale_raster( ( bathymetry - (NODATA_mask * datum) )[:,:ncols] ), # 20 m resolution
        nrows=int(nrows/2),
        ncols=int(ncols/2),
        cellsize=cellsize*2,
        filename="tauranga-20m.dem"
    )
    
    plot_bathymetry(
        bathymetry=( bathymetry - (NODATA_mask * datum) )[:,:ncols],
        nrows=nrows,
        ncols=ncols,
        xmin=0,
        ymin=0,
        cellsize=cellsize,
        filename="topography"
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
    
    write_stage_file()

def write_parameter_file(
    epsilon,
    solver,
    filename,
    results_dir
):
    params = (
        "test_case     0\n" +
        "max_ref_lvl   12\n" +
        "min_dt        1\n" +
        "respath       %s\n" +
        "epsilon       %s\n" +
        "fpfric        0.025\n" +
        "rasterroot    tauranga-10m\n" +
        "bcifile       tauranga.bci\n" +
        "bdyfile       tauranga.bdy\n" +
        "stagefile     tauranga.stage\n" +
        "tol_h         1e-2\n" +
        "tol_q         0\n" +
        "tol_s         1e-9\n" +
        "limitslopes   off\n" +
        "tol_Krivo     10\n" +
        "g             9.80665\n" +
        "massint       500\n" +
        "sim_time      144000\n" +
        "solver        %s\n" +
        "cumulative    on\n" +
        "voutput_stage on\n" +
        "wall_height   420"
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
    
    parameter_filename = "tauranga.par"
    
    write_parameter_file(
        epsilon=epsilon,
        solver=solver,
        filename=parameter_filename,
        results_dir=results_dir
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