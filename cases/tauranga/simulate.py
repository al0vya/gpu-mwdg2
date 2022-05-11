import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = (
        "Use this tool as follows:\n" +
        "python simulate.py preprocess\n" +
        "python simulate.py postprocess\n" +
        "python simulate.py simulate <SOLVER> <EPSILON>"
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

def write_bathymetry(
    bathymetry,
    nrows,
    ncols
):
    print("Preparing DEM file...")
    
    header = (
        "ncols        %s\n" +
        "nrows        %s\n" +
        "xllcorner    0\n" +
        "yllcorner    0\n" +
        "cellsize     10\n" +
        "NODATA_value -9999"
    ) % (
        ncols,
        nrows
    )
    
    np.savetxt(
        fname="tauranga-20m.dem",
        X=np.flipud(bathymetry),
        fmt="%.8f",
        header=header,
        comments=""
    )
    
def write_bci_file(
    ncols,
    cellsize,
    timeseries_name
):
    print("Preparing bci file...")
    
    xmax = ncols * cellsize
    
    with open("tauranga.bci", 'w') as fp:
        fp.write( "N 0 40960 HVAR %s" % (xmax, timeseries_name) )
        
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
    
    nrows, ncols = bathymetry.shape
    
    write_bathymetry(
        bathymetry=( bathymetry - (NODATA_mask * datum) ), # adjust datum only for non-NODATA_values
        nrows=nrows,
        ncols=ncols
    )
    
    cellsize = 10
    
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

def load_computed_gauge_timeseries():
    print("Loading computed gauges timeseries...")
    
    gauges = np.loadtxt(fname=os.path.join("results", "saved-stage-mwdg2.wd"), skiprows=11, delimiter=" ")
    
    datum = find_datum()
    
    return {
        "time"          : gauges[:,0] / 3600, # get into hours
        "A_Beacon"      : gauges[:,1] + datum,
        "Tug_Harbour"   : gauges[:,2] + datum,
        "Sulphur_Point" : gauges[:,3] + datum,
        "Moturiki"      : gauges[:,4] + datum
    }

def compare_timeseries(
    computed_gauges,
    experimental_gauges,
    name
):
    elevations = {
        "A_Beacon"      : 15.423466,
        "Tug_Harbour"   : 25.295409,
        "Sulphur_Point" : 25.272322,
        "Moturiki"      : 34.403634
    }
    
    fig, ax = plt.subplots()
    
    ax.plot(
        computed_gauges["time"],
        computed_gauges[name] + elevations[name],
        experimental_gauges[name]["time"],
        experimental_gauges[name]["total"]
    )
    
    ylim = (
        np.min( experimental_gauges[name]["total"] ),
        np.max( experimental_gauges[name]["total"] )
    )
    
    plt.setp(
        ax,
        title=name,
        xlim=( computed_gauges["time"][0], computed_gauges["time"][-1] ),
        ylim=ylim,
        xlabel=r"$t \, (hr)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    fig.savefig(os.path.join("results", name), bbox_inches="tight")
    
    plt.close()
    
def compare_all_timeseries():
    computed_gauges     = load_computed_gauge_timeseries()
    experimental_gauges = load_experimental_gauge_timeseries()
    
    compare_timeseries(computed_gauges=computed_gauges, experimental_gauges=experimental_gauges, name="A_Beacon")
    compare_timeseries(computed_gauges=computed_gauges, experimental_gauges=experimental_gauges, name="Tug_Harbour")
    compare_timeseries(computed_gauges=computed_gauges, experimental_gauges=experimental_gauges, name="Sulphur_Point")
    compare_timeseries(computed_gauges=computed_gauges, experimental_gauges=experimental_gauges, name="Moturiki")

def write_parameter_file(
    epsilon,
    solver,
    filename
):
    params = (
        "test_case   0\n" +
        "max_ref_lvl 11\n" +
        "min_dt      1\n" +
        "respath     results\n" +
        "epsilon     %s\n" +
        "fpfric      0.01\n" +
        "rasterroot  tauranga\n" +
        "bcifile     tauranga.bci\n" +
        "bdyfile     tauranga.bdy\n" +
        "stagefile   tauranga.stage\n" +
        "tol_h       1e-3\n" +
        "tol_q       0\n" +
        "tol_s       1e-9\n" +
        "limitslopes on\n" +
        "tol_Krivo   10\n" +
        "g           9.80665\n" +
        "saveint     3600\n" +
        "massint     500\n" +
        "sim_time    144000\n" +
        "solver      %s\n" +
        "cumulative  on\n" +
        "raster_out  on\n" +
        "wall_height 420"
    ) % (
        epsilon,
        solver
    )
    
    with open(filename, 'w') as fp:
        fp.write(params)

def run_simulation():
    if len(sys.argv) < 4: EXIT_HELP()
    
    dummy, option, solver, epsilon = sys.argv
    
    parameter_filename = "tauranga.par"
    
    write_parameter_file(
        epsilon=epsilon,
        solver=solver,
        filename=parameter_filename
    )
    
    subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), parameter_filename] )

if __name__ == "__main__":
    if len(sys.argv) < 2: EXIT_HELP()
    
    option = sys.argv[1]
    
    if   option == "preprocess":
        write_all_input_files()
    elif option == "postprocess":
        compare_all_timeseries()
    elif option == "simulate":
        run_simulation()
    else:
        EXIT_HELP()