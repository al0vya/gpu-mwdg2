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
        "36\n" +
        " 5.000   0.000\n" + # boundary cell
        "20.680  -0.515\n" + # W1
        "20.680   4.605\n" + # W2
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
    
    experimental_gauges["W1"] = {}
    experimental_gauges["W2"] = {}
    experimental_gauges["W3"] = {}
    experimental_gauges["W4"] = {}
    experimental_gauges["A1"] = {}
    experimental_gauges["A2"] = {}
    experimental_gauges["A3"] = {}
    experimental_gauges["A4"] = {}
    experimental_gauges["A5"] = {}
    experimental_gauges["A6"] = {}
    experimental_gauges["A7"] = {}
    experimental_gauges["A8"] = {}
    experimental_gauges["A9"] = {}
    experimental_gauges["B1"] = {}
    experimental_gauges["B2"] = {}
    experimental_gauges["B3"] = {}
    experimental_gauges["B4"] = {}
    experimental_gauges["B5"] = {}
    experimental_gauges["B6"] = {}
    experimental_gauges["B7"] = {}
    experimental_gauges["B8"] = {}
    experimental_gauges["B9"] = {}
    experimental_gauges["C1"] = {}
    experimental_gauges["C2"] = {}
    experimental_gauges["C3"] = {}
    experimental_gauges["C4"] = {}
    experimental_gauges["C5"] = {}
    experimental_gauges["C6"] = {}
    experimental_gauges["C7"] = {}
    experimental_gauges["C8"] = {}
    experimental_gauges["C9"] = {}
    experimental_gauges["D1"] = {}
    experimental_gauges["D2"] = {}
    experimental_gauges["D3"] = {}
    experimental_gauges["D4"] = {}
    
    wavegage = np.loadtxt(fname=os.path.join("comparison-data", "Wavegage.txt"), skiprows=1)
    
    A1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A1.txt"), skiprows=3)
    A2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A2.txt"), skiprows=3)
    A3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A3.txt"), skiprows=3)
    A4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A4.txt"), skiprows=3)
    A5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A5.txt"), skiprows=3)
    A6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A6.txt"), skiprows=3)
    A7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A7.txt"), skiprows=3)
    A8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A8.txt"), skiprows=3)
    A9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_A9.txt"), skiprows=3)
    B1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B1.txt"), skiprows=3)
    B2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B2.txt"), skiprows=3)
    B3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B3.txt"), skiprows=3)
    B4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B4.txt"), skiprows=3)
    B5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B5.txt"), skiprows=3)
    B6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B6.txt"), skiprows=3)
    B7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B7.txt"), skiprows=3)
    B8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B8.txt"), skiprows=3)
    B9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_B9.txt"), skiprows=3)
    C1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C1.txt"), skiprows=3)
    C2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C2.txt"), skiprows=3)
    C3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C3.txt"), skiprows=3)
    C4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C4.txt"), skiprows=3)
    C5 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C5.txt"), skiprows=3)
    C6 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C6.txt"), skiprows=3)
    C7 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C7.txt"), skiprows=3)
    C8 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C8.txt"), skiprows=3)
    C9 = np.loadtxt(fname=os.path.join("comparison-data", "Location_C9.txt"), skiprows=3)
    D1 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D1.txt"), skiprows=3)
    D2 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D2.txt"), skiprows=3)
    D3 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D3.txt"), skiprows=3)
    D4 = np.loadtxt(fname=os.path.join("comparison-data", "Location_D4.txt"), skiprows=3)
    
    experimental_gauges["W1"]["time"]    = wavegage[:,0]
    experimental_gauges["W2"]["time"]    = wavegage[:,0]
    experimental_gauges["W3"]["time"]    = wavegage[:,0]
    experimental_gauges["W4"]["time"]    = wavegage[:,0]
    
    experimental_gauges["W1"]["history"] = wavegage[:,3]
    experimental_gauges["W2"]["history"] = wavegage[:,4]
    experimental_gauges["W3"]["history"] = wavegage[:,5]
    experimental_gauges["W4"]["history"] = wavegage[:,6]
    
    experimental_gauges["A1"]["time"] = A1[:,0]
    experimental_gauges["A2"]["time"] = A2[:,0]
    experimental_gauges["A3"]["time"] = A3[:,0]
    experimental_gauges["A4"]["time"] = A4[:,0]
    experimental_gauges["A5"]["time"] = A5[:,0]
    experimental_gauges["A6"]["time"] = A6[:,0]
    experimental_gauges["A7"]["time"] = A7[:,0]
    experimental_gauges["A8"]["time"] = A8[:,0]
    experimental_gauges["A9"]["time"] = A9[:,0]
    experimental_gauges["B1"]["time"] = B1[:,0]
    experimental_gauges["B2"]["time"] = B2[:,0]
    experimental_gauges["B3"]["time"] = B3[:,0]
    experimental_gauges["B4"]["time"] = B4[:,0]
    experimental_gauges["B5"]["time"] = B5[:,0]
    experimental_gauges["B6"]["time"] = B6[:,0]
    experimental_gauges["B7"]["time"] = B7[:,0]
    experimental_gauges["B8"]["time"] = B8[:,0]
    experimental_gauges["B9"]["time"] = B9[:,0]
    experimental_gauges["C1"]["time"] = C1[:,0]
    experimental_gauges["C2"]["time"] = C2[:,0]
    experimental_gauges["C3"]["time"] = C3[:,0]
    experimental_gauges["C4"]["time"] = C4[:,0]
    experimental_gauges["C5"]["time"] = C5[:,0]
    experimental_gauges["C6"]["time"] = C6[:,0]
    experimental_gauges["C7"]["time"] = C7[:,0]
    experimental_gauges["C8"]["time"] = C8[:,0]
    experimental_gauges["C9"]["time"] = C9[:,0]
    experimental_gauges["D1"]["time"] = D1[:,0]
    experimental_gauges["D2"]["time"] = D2[:,0]
    experimental_gauges["D3"]["time"] = D3[:,0]
    experimental_gauges["D4"]["time"] = D4[:,0]
    
    experimental_gauges["A1"]["history"] = A1[:,1]
    experimental_gauges["A2"]["history"] = A2[:,1]
    experimental_gauges["A3"]["history"] = A3[:,1]
    experimental_gauges["A4"]["history"] = A4[:,1]
    experimental_gauges["A5"]["history"] = A5[:,1]
    experimental_gauges["A6"]["history"] = A6[:,1]
    experimental_gauges["A7"]["history"] = A7[:,1]
    experimental_gauges["A8"]["history"] = A8[:,1]
    experimental_gauges["A9"]["history"] = A9[:,1]
    experimental_gauges["B1"]["history"] = B1[:,1]
    experimental_gauges["B2"]["history"] = B2[:,1]
    experimental_gauges["B3"]["history"] = B3[:,1]
    experimental_gauges["B4"]["history"] = B4[:,1]
    experimental_gauges["B5"]["history"] = B5[:,1]
    experimental_gauges["B6"]["history"] = B6[:,1]
    experimental_gauges["B7"]["history"] = B7[:,1]
    experimental_gauges["B8"]["history"] = B8[:,1]
    experimental_gauges["B9"]["history"] = B9[:,1]
    experimental_gauges["C1"]["history"] = C1[:,1]
    experimental_gauges["C2"]["history"] = C2[:,1]
    experimental_gauges["C3"]["history"] = C3[:,1]
    experimental_gauges["C4"]["history"] = C4[:,1]
    experimental_gauges["C5"]["history"] = C5[:,1]
    experimental_gauges["C6"]["history"] = C6[:,1]
    experimental_gauges["C7"]["history"] = C7[:,1]
    experimental_gauges["C8"]["history"] = C8[:,1]
    experimental_gauges["C9"]["history"] = C9[:,1]
    experimental_gauges["D1"]["history"] = D1[:,1]
    experimental_gauges["D2"]["history"] = D2[:,1]
    experimental_gauges["D3"]["history"] = D3[:,1]
    experimental_gauges["D4"]["history"] = D4[:,1]
    
    return experimental_gauges

def load_computed_gauge_timeseries(
    stagefile
):
    print("Loading computed gauges timeseries: %s..." % stagefile)
    
    gauges = np.loadtxt(os.path.join("results", stagefile), skiprows=42, delimiter=" ")
    
    datum = -0.00202286243437291
    
    return {
        "time" : gauges[:,0],
        "BD"   : gauges[:,1]  + datum,
        "W1"   : gauges[:,2]  + datum,
        "W2"   : gauges[:,3]  + datum,
        "W3"   : gauges[:,4]  + datum,
        "W4"   : gauges[:,5]  + datum,
        "A1"   : gauges[:,6]  + datum,
        "A2"   : gauges[:,7]  + datum,
        "A3"   : gauges[:,8]  + datum,
        "A4"   : gauges[:,9]  + datum,
        "A5"   : gauges[:,10] + datum,
        "A6"   : gauges[:,11] + datum,
        "A7"   : gauges[:,12] + datum,
        "A8"   : gauges[:,13] + datum,
        "A9"   : gauges[:,14] + datum,
        "B1"   : gauges[:,15] + datum,
        "B2"   : gauges[:,16] + datum,
        "B3"   : gauges[:,17] + datum,
        "B4"   : gauges[:,18] + datum,
        "B5"   : gauges[:,19] + datum,
        "B6"   : gauges[:,20] + datum,
        "B7"   : gauges[:,21] + datum,
        "B8"   : gauges[:,22] + datum,
        "B9"   : gauges[:,23] + datum,
        "C1"   : gauges[:,24] + datum,
        "C2"   : gauges[:,25] + datum,
        "C3"   : gauges[:,26] + datum,
        "C4"   : gauges[:,27] + datum,
        "C5"   : gauges[:,28] + datum,
        "C6"   : gauges[:,29] + datum,
        "C7"   : gauges[:,30] + datum,
        "C8"   : gauges[:,31] + datum,
        "C9"   : gauges[:,32] + datum,
        "D1"   : gauges[:,33] + datum,
        "D2"   : gauges[:,34] + datum,
        "D3"   : gauges[:,35] + datum,
        "D4"   : gauges[:,36] + datum,
    }
    
def read_stage_elevations(
    stagefile
):
    header = []
    
    with open(os.path.join("results", stagefile), 'r') as fp:
        for i, line in enumerate(fp):
            if i > 2:
                header.append(line)
                
            if i > 38:
                break
    
    return {
        "BD" : float( header[0 ].split()[3] ),
        "W1" : float( header[1 ].split()[3] ),
        "W2" : float( header[2 ].split()[3] ),
        "W3" : float( header[3 ].split()[3] ),
        "W4" : float( header[4 ].split()[3] ),
        "A1" : float( header[5 ].split()[3] ),
        "A2" : float( header[6 ].split()[3] ),
        "A3" : float( header[7 ].split()[3] ),
        "A4" : float( header[8 ].split()[3] ),
        "A5" : float( header[9 ].split()[3] ),
        "A6" : float( header[10].split()[3] ),
        "A7" : float( header[11].split()[3] ),
        "A8" : float( header[12].split()[3] ),
        "A9" : float( header[13].split()[3] ),
        "B1" : float( header[14].split()[3] ),
        "B2" : float( header[15].split()[3] ),
        "B3" : float( header[16].split()[3] ),
        "B4" : float( header[17].split()[3] ),
        "B5" : float( header[18].split()[3] ),
        "B6" : float( header[19].split()[3] ),
        "B7" : float( header[20].split()[3] ),
        "B8" : float( header[21].split()[3] ),
        "B9" : float( header[22].split()[3] ),
        "C1" : float( header[23].split()[3] ),
        "C2" : float( header[24].split()[3] ),
        "C3" : float( header[25].split()[3] ),
        "C4" : float( header[26].split()[3] ),
        "C5" : float( header[27].split()[3] ),
        "C6" : float( header[28].split()[3] ),
        "C7" : float( header[29].split()[3] ),
        "C8" : float( header[30].split()[3] ),
        "C9" : float( header[31].split()[3] ),
        "D1" : float( header[32].split()[3] ),
        "D2" : float( header[33].split()[3] ),
        "D3" : float( header[34].split()[3] ),
        "D4" : float( header[35].split()[3] ),
    }

def compare_timeseries_stage(
    stagefiles,
    experimental_gauges,
    name
):
    print("Comparing timeseries at gauge %s" % name)
    
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
        #ylim=ylim,
        xlabel=r"$t \, (hr)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    plt.legend()
    
    fig.savefig(name, bbox_inches="tight")
    
    plt.close()
    
def compare_timeseries_all_stages():
    stagefiles = [
        #"stage-hwfv1-1e-3.wd",
        "stage.wd"
    ]
     
    experimental_gauges = load_experimental_gauge_timeseries()
    
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="W4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="A9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="B9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C4")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C5")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C6")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C7")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C8")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="C9")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D1")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D2")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D3")
    compare_timeseries_stage(stagefiles=stagefiles, experimental_gauges=experimental_gauges, name="D4")
    
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