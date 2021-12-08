import os
import sys

def EXIT_HELP():
    help_message = (
        "This tool is used in the command line as follows:\n\n" +
        " - python test.py run <MODE> <SOLVER> <EPSILON> <MAX_REF_LVL> (runs all in-built test cases)\n" +
        "    MODE        : [debug,release]\n" +
        "    SOLVER      : [hw,mw]\n" +
        "    EPSILON     : [error threshold]\n" +
        "    MAX_REF_LVL : [maximum refinment level]\n" +
        "\n" +
        " - python test.py soln <MODE> (plots solution contours)\n" +
        "    MODE : [debug,release]\n" +
        "\n" +
        " - python test.py c_prop <MODE> <SOLVER> <EPSILON> <MAX_REF_LVL> (plots discharge errors)\n" +
        "    MODE   : [debug,release]\n" +
        "    SOLVER : [hw,mw]"
    )

    sys.exit(help_message)

if len(sys.argv) < 3:
    EXIT_HELP()

import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab

from mpl_toolkits.mplot3d import Axes3D

def set_path(
    mode,
    testdir="test"
):
    if (mode == "debug"):
        path = os.path.join(os.path.dirname(__file__), "..", "out", "build", "x64-Debug", testdir, "results")
    elif (mode == "release"):
        path = os.path.join(os.path.dirname(__file__), "..", "out", "build", "x64-Release", testdir, "results")
    else:
        EXIT_HELP()
        
    return path

def plot_surface(
    X, 
    Y, 
    Z, 
    zlabel, 
    test_number, 
    path, 
    quantity, 
    test_name
):
    fig, ax = plt.subplots( subplot_kw={"projection" : "3d"} )
    
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel(zlabel)
    
    filename = str(test_number) + "-surf-" + quantity + "-" + test_name

    plt.savefig(os.path.join(path, filename), bbox_inches="tight")

    plt.clf()

def plot_contours(
    X, 
    Y, 
    Z, 
    ylabel, 
    test_number, 
    path, 
    quantity, 
    test_name
):
    fig, ax = plt.subplots()
    
    contourset = ax.contourf(X, Y, Z)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    
    colorbar = fig.colorbar(contourset)
    colorbar.ax.set_ylabel(ylabel)
    
    filename = str(test_number) + "-cont-" + quantity + "-" + test_name

    plt.savefig(os.path.join(path, filename), bbox_inches="tight")
    
    plt.clf()

class Solution:
    def __init__(
        self, 
        mode
    ):
        self.savepath = set_path(mode)
        
        print("Searching for solution data in path", self.savepath)
        
        h_file  = "depths.csv"
        qx_file = "discharge_x.csv"
        qy_file = "discharge_y.csv"
        z_file  = "topo.csv"
        
        # finest resolution mesh info
        mesh_info_file = "mesh_info.csv"
        
        mesh_info = pd.read_csv( os.path.join(self.savepath, mesh_info_file) )
        
        # to access a dataframe with only one row
        # we use iloc, which stands for 'integer location'
        mesh_dim = int(mesh_info.iloc[0][r"mesh_dim"])
        xsz      = int(mesh_info.iloc[0][r"xsz"])
        ysz      = int(mesh_info.iloc[0][r"ysz"])
        
        xmin = mesh_info.iloc[0]["xmin"]
        xmax = mesh_info.iloc[0]["xmax"]
        ymin = mesh_info.iloc[0]["ymin"]
        ymax = mesh_info.iloc[0]["ymax"]
        
        x_dim = xmax - xmin
        y_dim = ymax - ymin
        
        N_x = 1
        N_y = 1
        
        if (xsz >= ysz):
            N_x = xsz / ysz
        else:
            N_y = ysz / xsz
        
        x = np.linspace(xmin, xmax, xsz)
        y = np.linspace(ymin, ymax, ysz)
        
        self.X, self.Y = np.meshgrid(x, y)
        
        self.h  = pd.read_csv( os.path.join(self.savepath, h_file ) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.qx = pd.read_csv( os.path.join(self.savepath, qx_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.qy = pd.read_csv( os.path.join(self.savepath, qy_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.z  = pd.read_csv( os.path.join(self.savepath, z_file ) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]

    def plot_soln(
        self, 
        test_number=0, 
        test_name="ad-hoc"
    ):
        print("Plotting flow solution and topography for test %s..." % test_name)

        #plot_surface (self.X, self.Y, self.h, "$\eta \, (m)$",         test_number, self.path, "eta", test_name)
        #plot_surface (self.X, self.Y, self.qx,         "$qx \, (m^2s^{-1})$", test_number, self.path, "qx",  test_name)
        #plot_surface (self.X, self.Y, self.qy,         "$qy \, (m^2s^{-1})$", test_number, self.path, "qy",  test_name)
        
        plot_contours(self.X, self.Y, self.h,  "$h  \, (m)$",         test_number, self.savepath, "h",  test_name)
        plot_contours(self.X, self.Y, self.qx, "$qx \, (m^2s^{-1})$", test_number, self.savepath, "qx", test_name)
        plot_contours(self.X, self.Y, self.qy, "$qy \, (m^2s^{-1})$", test_number, self.savepath, "qy", test_name)
        plot_contours(self.X, self.Y, self.z,  "$z  \, (m)$",         test_number, self.savepath, "z",  test_name)

class DischargeErrors:
    def __init__(
        self, 
        solver, 
        mode
    ):
        self.solver = solver;
    
        self.savepath = set_path(mode)
        
        print("Searching for discharge error data in path", self.savepath)
        
        sim_time_file = "clock_time_vs_sim_time.csv"
        qx0_file      = "qx0-c-prop.csv"
        qx1x_file     = "qx1x-c-prop.csv"
        qx1y_file     = "qx1y-c-prop.csv"
        qy0_file      = "qy0-c-prop.csv"
        qy1x_file     = "qy1x-c-prop.csv"
        qy1y_file     = "qy1y-c-prop.csv"
        
        self.sim_time = pd.read_csv( os.path.join(self.savepath, sim_time_file) )
        qx0           = pd.read_csv( os.path.join(self.savepath, qx0_file),  header=None )
        qx1x          = pd.read_csv( os.path.join(self.savepath, qx1x_file), header=None ) if solver == "mw" else None
        qx1y          = pd.read_csv( os.path.join(self.savepath, qx1y_file), header=None ) if solver == "mw" else None
        qy0           = pd.read_csv( os.path.join(self.savepath, qy0_file),  header=None )
        qy1x          = pd.read_csv( os.path.join(self.savepath, qy1x_file), header=None ) if solver == "mw" else None
        qy1y          = pd.read_csv( os.path.join(self.savepath, qy1y_file), header=None ) if solver == "mw" else None
        
        self.qx0_max  = qx0.abs().max(axis=1)
        self.qx1x_max = qx1x.abs().max(axis=1) if solver == "mw" else None
        self.qx1y_max = qx1y.abs().max(axis=1) if solver == "mw" else None
        self.qy0_max  = qy0.abs().max(axis=1)
        self.qy1x_max = qy1x.abs().max(axis=1) if solver == "mw" else None
        self.qy1y_max = qy1y.abs().max(axis=1) if solver == "mw" else None

    def plot_errors(
        self, 
        test_number=0, 
        test_name="ad-hoc"
    ):

        print("Plotting maximum discharge errors for test %s..." % test_name)

        plt.figure()
        
        plt.scatter(self.sim_time["sim_time"], self.qx0_max,  label='$q^0_x$', marker='x')
        plt.scatter(self.sim_time["sim_time"], self.qy0_max,  label='$q^0_y$', marker='x')
        
        if self.solver == "mw":
            plt.scatter(self.sim_time["sim_time"], self.qx1x_max, label='$q^{1x}_x$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qx1y_max, label='$q^{1y}_x$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qy1x_max, label='$q^{1x}_y$', marker='x')
            plt.scatter(self.sim_time["sim_time"], self.qy1y_max, label='$q^{1y}_y$', marker='x')
        
        xlim = ( self.sim_time["sim_time"].iloc[0], self.sim_time["sim_time"].iloc[-1] )

        plt.ticklabel_format(axis='x', style="sci")
        plt.xlim(xlim)
        plt.legend()
        plt.ylabel("Maximum error")
        plt.xlabel("Simulation time (s)")

        filename = str(test_number) + "-c-prop-" + test_name

        plt.savefig(os.path.join(self.savepath, filename), bbox_inches="tight")
        
        plt.clf()

class Test:
    def __init__(
        self,
        test_case, 
        max_ref_lvl, 
        epsilon, 
        massint, 
        test_name, 
        solver, 
        c_prop_tests, 
        results, 
        input_file,
        mode
    ):
        self.test_case   = test_case
        self.max_ref_lvl = max_ref_lvl
        self.epsilon     = epsilon
        self.massint     = massint 
        self.test_name   = test_name
        self.solver      = solver

        if self.test_case in c_prop_tests:
            self.row_major = "off"
            self.vtk       = "off"
            self.c_prop    = "on"
        else:
            self.row_major = "on"
            self.vtk       = "on"
            self.c_prop    = "off"

        self.results    = results
        self.input_file = input_file
        self.mode       = mode

    def set_params(
        self
    ):
        params = ("" +
            "test_case   %s\n" +
            "max_ref_lvl	%s\n" +
            "min_dt		1\n" +
            "respath	    %s\n" +
            "epsilon	    %s\n" +
            "tol_h		1e-4\n" +
            "tol_q		0\n" +
            "tol_s		1e-9\n" +
            "g			9.80665\n" +
            "massint		%s\n" +
            "solver		%s\n" +
            "wall_height	0\n" +
            "row_major    %s\n" +
            "c_prop %s\n" +
            "vtk        %s") % (
                self.test_case, 
                self.max_ref_lvl, 
                self.results, 
                self.epsilon, 
                self.massint, 
                self.solver, 
                self.row_major, 
                self.c_prop, 
                self.vtk
            )

        with open(self.input_file, 'w') as fp:
            fp.write(params)

    def run_test(
        self,
        solver_file
    ):
        self.set_params()

        subprocess.run( [solver_file, self.input_file] )

        if self.c_prop == "on":
            DischargeErrors(self.solver, self.mode).plot_errors(self.test_case, self.test_name)
        else:
            Solution(self.mode).plot_soln(self.test_case, self.test_name)

def plot_soln():
    if len(sys.argv) > 2:
        mode = sys.argv[2]
        
        if mode == "debug" or mode == "release":
            Solution(mode).plot_soln()
        else:
            EXIT_HELP()
    else:
        EXIT_HELP()

def plot_c_prop():
    if len(sys.argv) > 3:
        dummy, mode, solver = sys.argv
        
        if mode == "debug" or mode == "release":
            if solver == "hw" or solver == "mw":
                DischargeErrors(solver, mode).plot_errors()
            else:
                EXIT_HELP()
        else:
            EXIT_HELP()
    else:
        EXIT_HELP()

def run_tests():
    if len(sys.argv) > 5:
        dummy, action, mode, solver, epsilon, max_ref_lvl = sys.argv
    
        if mode == "debug":
            path = os.path.join("..", "out", "build", "x64-Debug")
        elif mode == "release":
            path = os.path.join("..", "out", "build", "x64-Release")
        else:
            EXIT_HELP()
            
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
    else:
        EXIT_HELP()

    input_file  = os.path.join(path, "test", "inputs.par")
    solver_file = os.path.join(path, "gpu-mwdg2.exe")
    results     = os.path.join(path, "test", "results")
    
    test_names = [
    "1D-c-prop-y-dir-wet",
    "1D-c-prop-x-dir-wet",
    "1D-c-prop-x-dir-wet-dry",
    "1D-c-prop-y-dir-wet-dry",
    "wet-dam-break-x-dir",
    "wet-dam-break-y-dir",
    "dry-dam-break-x-dir",
    "dry-dam-break-y-dir",
    "dry-dam-break-w-fric-x-dir",
    "dry-dam-break-w-fric-y-dir",
    "wet-building-overtopping-x-dir",
    "wet-building-overtopping-y-dir",
    "dry-building-overtopping-x-dir",
    "dry-building-overtopping-y-dir",
    "triangular-dam-break-x-dir",
    "triangular-dam-break-y-dir",
    "parabolic-bowl-x-dir",
    "parabolic-bowl-y-dir",
    "three-cones",
    "differentiable-blocks",
    "non-differentiable-blocks",
    "radial-dam-break"
    ]
    
    massints = [
    1,     # 1D c prop x dir
    1,     # 1D c prop y dir
    1,     # 1D c prop x dir
    1,     # 1D c prop y dir
    2.5,   # wet dam break x dir
    2.5,   # wet dam break y dir
    1.3,   # dry dam break x dir
    1.3,   # dry dam break y dir
    1.3,   # dry dam break wh fric x dir
    1.3,   # dry dam break wh fric y dir
    10,    # wet building overtopping x dir
    10,    # wet building overtopping y dir
    10,    # dry building overtopping x dir
    10,    # dry building overtopping y dir
    29.6,  # triangular dam break x dir
    29.6,  # triangular dam break y dir
    21.6,  # parabolic bowl x dir
    21.6,  # parabolic bowl y dir
    1,     # three cones
    1,     # differentiable blocks
    1,     # non-differentiable blocks
    3.5    # radial dam break
    ]
    
    num_tests = 22
    
    c_prop_tests = [1, 2, 3, 4, 19, 20, 21]
    
    for test in range(num_tests):
        Test(
            test + 1,
            max_ref_lvl,
            epsilon,
            massints[test],
            test_names[test],
            solver,
            c_prop_tests,
            results,
            input_file,
            mode
        ).run_test(solver_file)

# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
params = {
'legend.fontsize' : 'xx-large',
'axes.labelsize'  : 'xx-large',
'axes.titlesize'  : 'xx-large',
'xtick.labelsize' : 'xx-large',
'ytick.labelsize' : 'xx-large'
}

pylab.rcParams.update(params)

action = sys.argv[1]

if   action == "run":
    run_tests()
elif action == "soln":
    plot_soln()
elif action == "c_prop":
    plot_c_prop()
else:
    EXIT_HELP()