import os
import sys

def EXIT_HELP():
    help_message = (
        "This tool is used in the command line as follows:\n\n" +
        " - python test.py test <SOLVER> <EPSILON> <MAX_REF_LVL> <PLOT_TYPE> (runs all in-built test cases)\n" +
        "    SOLVER      : [hw,mw]\n" +
        "    EPSILON     : [error threshold]\n" +
        "    MAX_REF_LVL : [maximum refinment level]\n" +
        "    PLOT_TYPE   : [cont,surf]\n" +
            "\n" +
        " - python test.py run <SOLVER> <TEST_CASE> <SIM_TIME> <EPSILON> <MAX_REF_LVL> <SAVE_INT> <MASS_INT> <PLOT_TYPE> (runs a single in-built test cases)\n" +
        "    SOLVER      : [hw,mw]\n" +
        "    TEST_CASE   : [1-22 inclusive]\n" +
        "    SIM_TIME    : [simulation time in seconds]\n" +
        "    EPSILON     : [error threshold]\n" +
        "    MAX_REF_LVL : [maximum refinment level]\n" +
        "    SAVE_INT    : [interval in seconds that solution data are saved]\n" +
        "    MASS_INT    : [interval in seconds that simulation times are saved]\n" +
        "    PLOT_TYPE   : [cont,surf]\n" +
        "\n" +
        " - python test.py planar <SOLVER> <TEST_CASE_DIR> <PHYS_QUANTITY> <INTERVAL> (plots planar solution)\n" +
        "    SOLVER        : [hw,mw]\n" +
        "    PHYS_QUANTITY : [h,eta,qx,qy,z]\n" +
        "    INTERVAL      : [interval]\n" +
        "\n" +
        " - python test.py row_major <PLOT_TYPE> (plots either solution surface or contours)\n" +
        "    PLOT_TYPE : [cont,surf]\n" +
        "\n" +
        " - python test.py c_prop <SOLVER> (plots discharge errors)\n" +
        "    SOLVER : [hw,mw]"
    )

    sys.exit(help_message)

if len(sys.argv) < 3:
    EXIT_HELP()

import imageio
import subprocess
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab

from mpl_toolkits.mplot3d import Axes3D

####################################
# NATURAL SORTING, READ UP ON THIS #
####################################

import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def get_filenames_natural_order(path):
    filenames_natural_order = os.listdir(path)
    
    filenames_natural_order.sort(key=natural_keys)
    
    return filenames_natural_order

####################################
####################################
####################################

def clear_files(
    path,
    file_extension
):
    print("Clearing %s files..." % file_extension)
    
    for filename in os.listdir(path):
        if filename.endswith("." + file_extension):
            os.remove( os.path.join(path, filename) )

test_names = [
    "1D-c-prop-x-dir-wet",
    "1D-c-prop-y-dir-wet",
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
    "three-cones-dam-break",
    "differentiable-blocks",
    "non-differentiable-blocks",
    "radial-dam-break"
]

c_prop_tests = [1, 2, 3, 4, 21, 22]

sim_times = [
    80,   # 1D c prop x dir
    80,   # 1D c prop y dir
    80,   # 1D c prop x dir
    80,   # 1D c prop y dir
    2.5,  # wet dam break x dir
    2.5,  # wet dam break y dir
    1.3,  # dry dam break x dir
    1.3,  # dry dam break y dir
    1.3,  # dry dam break w fric x dir
    1.3,  # dry dam break w fric y dir
    10,   # wet building overtopping x dir
    10,   # wet building overtopping y dir
    10,   # dry building overtopping x dir
    10,   # dry building overtopping y dir
    29.6, # triangular dam break x dir
    29.6, # triangular dam break y dir
    21.6, # parabolic bowl x dir
    21.6, # parabolic bowl y dir
    80,   # three cones
    30,   # three cones dam break
    80,   # differentiable blocks
    80,   # non-differentiable blocks
    3.5   # radial dam break
]

class PlanarSolution:
    def __init__(
        self,
        solver,
        interval
    ):
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
            
        self.solver = solver
        
        self.interval = interval
        
        self.savepath = os.path.join("results", "planar-" + str(self.interval) + ".csv")
        
        planar_dataframe = pd.read_csv(self.savepath)
        
        self.lower_left_x = planar_dataframe["lower_left_x"].values
        self.lower_left_y = planar_dataframe["lower_left_y"].values
        
        self.upper_right_x = planar_dataframe["upper_right_x"].values
        self.upper_right_y = planar_dataframe["upper_right_y"].values
        
        self.h0   = planar_dataframe["h0"].values
        self.h1x  = planar_dataframe["h1x"].values  if solver == "mw" else None
        self.h1y  = planar_dataframe["h1y"].values  if solver == "mw" else None
        
        self.qx0  = planar_dataframe["qx0"].values
        self.qx1x = planar_dataframe["qx1x"].values if solver == "mw" else None
        self.qx1y = planar_dataframe["qx1y"].values if solver == "mw" else None
        
        self.qy0  = planar_dataframe["qy0"].values
        self.qy1x = planar_dataframe["qy1x"].values if solver == "mw" else None
        self.qy1y = planar_dataframe["qy1y"].values if solver == "mw" else None
        
        self.z0   = planar_dataframe["z0"].values
        self.z1x  = planar_dataframe["z1x"].values  if solver == "mw" else None
        self.z1y  = planar_dataframe["z1y"].values  if solver == "mw" else None
        
        self.num_cells = self.h0.size
        
        self.fig, self.ax = plt.subplots( subplot_kw={"projection" : "3d"} )
    
    def plot_soln(
        self,
        quantity
    ):
        print("Plotting planar solution...")

        S = None
        
        for cell in range(self.num_cells):
            print("Cell", cell + 1, "of", self.num_cells)
            
            x = [ self.lower_left_x[cell], self.upper_right_x[cell] ]
            y = [ self.lower_left_y[cell], self.upper_right_y[cell] ]
            
            X, Y = np.meshgrid(x, y)
            
            upper_left_h   = self.h0[cell]  - np.sqrt(3) * self.h1x[cell]  + np.sqrt(3) * self.h1y[cell]  if self.solver == "mw" else self.h0[cell]
            upper_right_h  = self.h0[cell]  + np.sqrt(3) * self.h1x[cell]  + np.sqrt(3) * self.h1y[cell]  if self.solver == "mw" else self.h0[cell]
            lower_left_h   = self.h0[cell]  - np.sqrt(3) * self.h1x[cell]  - np.sqrt(3) * self.h1y[cell]  if self.solver == "mw" else self.h0[cell]
            lower_right_h  = self.h0[cell]  + np.sqrt(3) * self.h1x[cell]  - np.sqrt(3) * self.h1y[cell]  if self.solver == "mw" else self.h0[cell]
            
            upper_left_qx  = self.qx0[cell] - np.sqrt(3) * self.qx1x[cell] + np.sqrt(3) * self.qx1y[cell] if self.solver == "mw" else self.qx0[cell]
            upper_right_qx = self.qx0[cell] + np.sqrt(3) * self.qx1x[cell] + np.sqrt(3) * self.qx1y[cell] if self.solver == "mw" else self.qx0[cell]
            lower_left_qx  = self.qx0[cell] - np.sqrt(3) * self.qx1x[cell] - np.sqrt(3) * self.qx1y[cell] if self.solver == "mw" else self.qx0[cell]
            lower_right_qx = self.qx0[cell] + np.sqrt(3) * self.qx1x[cell] - np.sqrt(3) * self.qx1y[cell] if self.solver == "mw" else self.qx0[cell]
            
            upper_left_qy  = self.qy0[cell] - np.sqrt(3) * self.qy1x[cell] + np.sqrt(3) * self.qy1y[cell] if self.solver == "mw" else self.qy0[cell]
            upper_right_qy = self.qy0[cell] + np.sqrt(3) * self.qy1x[cell] + np.sqrt(3) * self.qy1y[cell] if self.solver == "mw" else self.qy0[cell]
            lower_left_qy  = self.qy0[cell] - np.sqrt(3) * self.qy1x[cell] - np.sqrt(3) * self.qy1y[cell] if self.solver == "mw" else self.qy0[cell]
            lower_right_qy = self.qy0[cell] + np.sqrt(3) * self.qy1x[cell] - np.sqrt(3) * self.qy1y[cell] if self.solver == "mw" else self.qy0[cell]
            
            upper_left_z   = self.z0[cell]  - np.sqrt(3) * self.z1x[cell]  + np.sqrt(3) * self.z1y[cell]  if self.solver == "mw" else self.z0[cell]
            upper_right_z  = self.z0[cell]  + np.sqrt(3) * self.z1x[cell]  + np.sqrt(3) * self.z1y[cell]  if self.solver == "mw" else self.z0[cell]
            lower_left_z   = self.z0[cell]  - np.sqrt(3) * self.z1x[cell]  - np.sqrt(3) * self.z1y[cell]  if self.solver == "mw" else self.z0[cell]
            lower_right_z  = self.z0[cell]  + np.sqrt(3) * self.z1x[cell]  - np.sqrt(3) * self.z1y[cell]  if self.solver == "mw" else self.z0[cell]
            
            H  = np.asarray( [ [lower_left_h,  lower_right_h ], [upper_left_h,  upper_right_h ] ] )
            QX = np.asarray( [ [lower_left_qx, lower_right_qx], [upper_left_qx, upper_right_qx] ] )
            QY = np.asarray( [ [lower_left_qy, lower_right_qy], [upper_left_qy, upper_right_qy] ] )
            Z  = np.asarray( [ [lower_left_z,  lower_right_z ], [upper_left_z,  upper_right_z ] ] )
            
            if   quantity == 'h':
                S = H
            elif quantity == "eta":
                S = H + Z
            elif quantity == "qx":
                S = QX
            elif quantity == "qy":
                S = QY
            elif quantity == 'z':
                S = Z
            else:
                EXIT_HELP()
            
            self.ax.plot_surface(X, Y, S, color="#599DEE", rcount=1, ccount=1, shade=False, edgecolors='k', linewidth=0.25)
            
        elev = 29   if quantity != 'h' else 52
        azim = -120 if quantity != 'h' else 40
        
        self.ax.view_init(elev, azim)
        plt.savefig(os.path.join(self.savepath, "planar-soln-" + str(self.interval) + ".svg"), bbox_inches="tight")

class Limits:
    def __init__(
            self,
            intervals
        ):
            h  = []
            qx = []
            qy = []
            z  = []
            
            for interval in range(intervals + 1):
                h_file  = "depths-" +      str(interval) + ".csv"
                qx_file = "discharge_x-" + str(interval) + ".csv"
                qy_file = "discharge_y-" + str(interval) + ".csv"
                z_file  = "topo-" +        str(interval) + ".csv"
                
                h  += pd.read_csv( os.path.join("results", h_file ) )["results"].to_list()
                qx += pd.read_csv( os.path.join("results", qx_file) )["results"].to_list()
                qy += pd.read_csv( os.path.join("results", qy_file) )["results"].to_list()
                z  += pd.read_csv( os.path.join("results", z_file ) )["results"].to_list()
                
            self.h_max  = np.max(h)
            self.qx_max = np.max(qx)
            self.qy_max = np.max(qy)
            self.z_max  = np.max(z)
            
            self.h_min  = np.min(h)
            self.qx_min = np.min(qx)
            self.qy_min = np.min(qy)
            self.z_min  = np.min(z)

def plot_surface(
    X,
    Y,
    Z,
    zlim,
    zlabel,
    test_number,
    quantity,
    interval,
    test_name
):
    fig, ax = plt.subplots( subplot_kw={"projection" : "3d"} )
    
    ax.plot_surface(X, Y, Z)
    ax.set_zlim(zlim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel(zlabel)
    
    filename = test_name + "-surf-" + str(interval) + "-" + quantity + ".jpg" 
    
    plt.savefig(os.path.join("results", filename), bbox_inches="tight")

    plt.close()

def plot_contours(
    X,
    Y,
    Z,
    zlim,
    ylabel,
    test_number,
    quantity,
    interval,
    test_name
):
    num_levels = 20
    dZ = ( zlim[1] - zlim[0] ) / num_levels
    
    Z_levels = [ zlim[0] + dZ * n for n in range(num_levels + 1) ]
    
    # ensure no array of zeroes because levels in contour plot must be increasing
    Z_levels = Z_levels if ( Z_levels[1] - Z_levels[0] ) > 0 else num_levels
    
    fig, ax = plt.subplots()
    
    contourset = ax.contourf(X, Y, Z, levels=Z_levels)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    
    colorbar = fig.colorbar(contourset)
    colorbar.ax.set_ylabel(ylabel)
    
    filename = test_name + "-cont-" + str(interval) + "-" + quantity + ".jpg"

    plt.savefig(os.path.join("results", filename), bbox_inches="tight")
    
    plt.close()

class RowMajorSolution:
    def __init__(
        self,
        interval
    ):
        self.interval = interval
        
        h_file  = "depths-" +      str(self.interval) + ".csv"
        qx_file = "discharge_x-" + str(self.interval) + ".csv"
        qy_file = "discharge_y-" + str(self.interval) + ".csv"
        z_file  = "topo-" +        str(self.interval) + ".csv"
        
        mesh_info = pd.read_csv( os.path.join("results", "mesh_info.csv") )
        
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
        
        self.h  = pd.read_csv( os.path.join("results", h_file ) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.qx = pd.read_csv( os.path.join("results", qx_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.qy = pd.read_csv( os.path.join("results", qy_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
        self.z  = pd.read_csv( os.path.join("results", z_file ) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]

    def plot_surfaces(
        self,
        limits,
        test_number=0, 
        test_name="ad-hoc"
    ):
        print("Plotting flow solution and topography for test %s..." % test_name)

        plot_surface(self.X, self.Y, self.h,  (limits.h_min,  limits.h_max),  "$h \, (m)$",           test_number, "h",  self.interval, test_name)
        plot_surface(self.X, self.Y, self.qx, (limits.qx_min, limits.qx_max), "$q_x \, (m^2s^{-1})$", test_number, "qx", self.interval, test_name)
        plot_surface(self.X, self.Y, self.qy, (limits.qy_min, limits.qy_max), "$q_y \, (m^2s^{-1})$", test_number, "qy", self.interval, test_name)

    def plot_contours(
        self,
        limits,
        test_number=0, 
        test_name="ad-hoc"
    ):
        print("Plotting flow solution and topography for test %s..." % test_name)
        
        plot_contours(self.X, self.Y, self.h,  (limits.h_min,  limits.h_max),  "$h  \, (m)$",          test_number, "h",  self.interval, test_name)
        plot_contours(self.X, self.Y, self.qx, (limits.qx_min, limits.qx_max), "$q_x \, (m^2s^{-1})$", test_number, "qx", self.interval, test_name)
        plot_contours(self.X, self.Y, self.qy, (limits.qy_min, limits.qy_max), "$q_y \, (m^2s^{-1})$", test_number, "qy", self.interval, test_name)
        plot_contours(self.X, self.Y, self.z,  (limits.z_min,  limits.z_max),  "$z  \, (m)$",          test_number, "z",  self.interval, test_name)
        
    def plot_soln(
        self,
        plot_type,
        limits=None,
        test_number=0,
        test_name="ad-hoc"
    ):
        params = {
            "legend.fontsize" : "xx-large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large"
        }
        
        plt.rcParams.update(params)
        
        if   plot_type == "cont":
            self.plot_contours(limits, test_number, test_name)
        elif plot_type == "surf":
            self.plot_surfaces(limits, test_number, test_name)
        else:
            EXIT_HELP()
        
class DischargeErrors:
    def __init__(
        self, 
        solver
    ):
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
        
        self.solver = solver
        
        simtime_file = "simtime-vs-runtime.csv"
        qx0_file     = "qx0-c-prop.csv"
        qx1x_file    = "qx1x-c-prop.csv"
        qx1y_file    = "qx1y-c-prop.csv"
        qy0_file     = "qy0-c-prop.csv"
        qy1x_file    = "qy1x-c-prop.csv"
        qy1y_file    = "qy1y-c-prop.csv"
        
        self.simtime = pd.read_csv( os.path.join("results", simtime_file) )
        qx0          = pd.read_csv( os.path.join("results", qx0_file),  header=None )
        qx1x         = pd.read_csv( os.path.join("results", qx1x_file), header=None ) if solver == "mw" else None
        qx1y         = pd.read_csv( os.path.join("results", qx1y_file), header=None ) if solver == "mw" else None
        qy0          = pd.read_csv( os.path.join("results", qy0_file),  header=None )
        qy1x         = pd.read_csv( os.path.join("results", qy1x_file), header=None ) if solver == "mw" else None
        qy1y         = pd.read_csv( os.path.join("results", qy1y_file), header=None ) if solver == "mw" else None
        
        self.qx0_max  = qx0.abs().max(axis=1)
        self.qx1x_max = qx1x.abs().max(axis=1) if solver == "mw" else None
        self.qx1y_max = qx1y.abs().max(axis=1) if solver == "mw" else None
        self.qy0_max  = qy0.abs().max(axis=1)
        self.qy1x_max = qy1x.abs().max(axis=1) if solver == "mw" else None
        self.qy1y_max = qy1y.abs().max(axis=1) if solver == "mw" else None

    def plot_errors(
        self, 
        test_name="ad-hoc"
    ):
        params = {
            "legend.fontsize" : "large",
            "axes.labelsize"  : "xx-large",
            "axes.titlesize"  : "xx-large",
            "xtick.labelsize" : "xx-large",
            "ytick.labelsize" : "xx-large"
        }
        
        plt.rcParams.update(params)
        
        print("Plotting maximum discharge errors for test %s..." % test_name)

        fig, ax = plt.subplots( figsize=(5, 2) )
        
        plt.scatter(self.simtime["simtime"], self.qx0_max,  label='$q^0_x$', marker='x')
        plt.scatter(self.simtime["simtime"], self.qy0_max,  label='$q^0_y$', marker='x')
        
        if self.solver == "mw":
            plt.scatter(self.simtime["simtime"], self.qx1x_max, label='$q^{1x}_x$', marker='x')
            plt.scatter(self.simtime["simtime"], self.qx1y_max, label='$q^{1y}_x$', marker='x')
            plt.scatter(self.simtime["simtime"], self.qy1x_max, label='$q^{1x}_y$', marker='x')
            plt.scatter(self.simtime["simtime"], self.qy1y_max, label='$q^{1y}_y$', marker='x')
        
        xlim = ( self.simtime["simtime"].iloc[0], self.simtime["simtime"].iloc[-1] )

        plt.ticklabel_format(axis='x', style="sci")
        plt.xlim(xlim)
        plt.ylim(0, 1e-10)
        #plt.yscale("log")
        plt.legend()
        plt.ylabel("Maximum error")
        plt.xlabel("Simulation time (s)")

        filename = "c-prop-" + test_name

        plt.savefig(os.path.join("results", filename), bbox_inches="tight")
        
        plt.close()

class Test:
    def __init__(
        self,
        test_case, 
        sim_time, 
        max_ref_lvl, 
        epsilon, 
        saveint, 
        massint, 
        test_name, 
        solver
    ):
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
        
        self.test_case   = test_case
        self.sim_time    = sim_time
        self.max_ref_lvl = max_ref_lvl
        self.epsilon     = epsilon
        self.saveint     = saveint 
        self.massint     = massint
        self.test_name   = test_name
        self.solver      = solver

        if self.test_case in c_prop_tests:
            self.row_major  = "off"
            self.vtk        = "off"
            self.c_prop     = "on"
            self.cumulative = "off"
        else:
            self.row_major  = "on"
            self.vtk        = "off"
            self.c_prop     = "off"
            self.cumulative = "on"

        self.input_file = "inputs.par"
        self.intervals  = int(self.sim_time / self.saveint)

    def set_params(
        self
    ):
        params = (
            "test_case   %s\n" +
            "sim_time    %s\n" +
            "max_ref_lvl %s\n" +
            "min_dt      0.5\n" +
            "respath     results\n" +
            "epsilon     %s\n" +
            "tol_h       1e-3\n" +
            "tol_q       0\n" +
            "tol_s       1e-9\n" +
            "g           9.80665\n" +
            "saveint     %s\n" +
            "massint     %s\n" +
            "solver      %s\n" +
            "wall_height 0\n" +
            "row_major   %s\n" +
            "c_prop      %s\n" +
            "cumulative  %s\n" +
            "limitslopes off\n" +
            "tol_Krivo   1\n" +
            "vtk         %s") % (
                self.test_case, 
                self.sim_time, 
                self.max_ref_lvl,
                self.epsilon, 
                self.saveint, 
                self.massint, 
                self.solver, 
                self.row_major, 
                self.c_prop, 
                self.cumulative,
                self.vtk
            )

        with open(self.input_file, 'w') as fp:
            fp.write(params)

    def run_test(
        self,
        plot_type
    ):
        self.set_params()
        
        subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), self.input_file] )

        if self.c_prop == "on":
            DischargeErrors(self.solver).plot_errors(self.test_name)
        else:
            limits=Limits(self.intervals)
            
            for interval in range(1, self.intervals + 1):
                RowMajorSolution(interval).plot_soln(
                    limits=limits,
                    test_number=self.test_case,
                    test_name=self.test_name,
                    plot_type=plot_type
                )

def animate(path):
    images = []
    
    filenames_natural_order = get_filenames_natural_order(path)
    
    vars = ["h", "qx", "qy"]
    
    for test_name in test_names:
        for var in vars:
            suffix = var + ".jpg"
            for filename in filenames_natural_order:
                if filename.startswith(test_name) and filename.endswith(suffix):
                    image = imageio.imread( os.path.join(path, filename) )
                    
                    images.append(image)
            
            if images:
                imageio.mimsave(os.path.join(path, test_name + "-" + var + ".gif"), images)
                images = []

def run():
    print("Attempting to run test...")
    
    if len(sys.argv) > 9:
        dummy, action, solver, test_case, sim_time, epsilon, max_ref_lvl, saveint, massint, plot_type = sys.argv
        
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
    else:
        EXIT_HELP()
    
    clear_files("results", "jpg")
    
    Test(
        test_case=int(test_case),
        sim_time=float(sim_time),
        max_ref_lvl=int(max_ref_lvl),
        epsilon=float(epsilon),
        saveint=float(saveint),
        massint=float(massint),
        test_name=test_names[int(test_case) - 1],
        solver=solver
    ).run_test(plot_type)
    
    animate("results")
    
    clear_files("results", "jpg")
    
def run_tests():
    print("Attempting to run tests specified in tests.txt, checking input parameters...")
    
    if len(sys.argv) > 5:
        dummy, action, solver, epsilon, max_ref_lvl, plot_type = sys.argv
        
        if solver != "hw" and solver != "mw":
            EXIT_HELP()
    else:
        EXIT_HELP()
        
    tests = []
    
    with open("tests.txt", 'r') as fp:
        tests = fp.readlines()
        tests = [int( test.rstrip() ) for test in tests]
   
    # make interval very large because not saving simulation times
    massint = 9999
    
    for test in tests:
        Test(
            test_case=int(test),
            sim_time=sim_times[test - 1],
            max_ref_lvl=int(max_ref_lvl),
            epsilon=float(epsilon),
            saveint=sim_times[test - 1] / 80 if test in c_prop_tests else sim_times[test - 1],
            massint=float(massint),
            test_name=test_names[test - 1],
            solver=solver
        ).run_test(plot_type)

def plot_soln_planar():
    if len(sys.argv) > 5:
        dummy, action, solver, results, quantity, interval = sys.argv
        
        PlanarSolution(solver, interval).plot_soln(quantity)
    else:
        EXIT_HELP()

def plot_soln_row_major():
    if len(sys.argv) > 4:
        dummy, action, interval, results, plot_type = sys.argv
        
        RowMajorSolution(interval).plot_soln(plot_type)
    else:
        EXIT_HELP()

def plot_c_prop():
    if len(sys.argv) > 3:
        dummy, action, results, solver = sys.argv
        
        DischargeErrors(solver).plot_errors()
    else:
        EXIT_HELP()

if __name__ == "__main__":
    
    action = sys.argv[1]
    
    #clear_files("results", "csv")
    #clear_files("results", "jpg")
    
    if   action == "test":
        run_tests()
    elif action == "run":
        run()
    elif action == "planar":
        plot_soln_planar()
    elif action == "row_major":
        plot_soln_row_major()
    elif action == "c_prop":
        plot_c_prop()
    else:
        EXIT_HELP
        
    #clear_files("results", "csv")
    #clear_files("results", "jpg")