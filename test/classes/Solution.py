import os
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

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
        relativepath=""
    ):
        self.relativepath = relativepath

        if (relativepath == "debug"):
            self.relativepath = os.path.join("..", "..", "out", "build", "x64-Debug", "test", "results")
        elif (relativepath == "release"):
            self.relativepath = os.path.join("..", "..", "out", "build", "x64-Release", "test", "results")
        
        self.savepath = os.path.join(os.path.dirname(__file__), self.relativepath)
 
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