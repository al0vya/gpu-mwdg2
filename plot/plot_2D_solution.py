import os
import matplotlib.pyplot as plt
import matplotlib.pylab  as pylab
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys

from mpl_toolkits.mplot3d import Axes3D # import for 3D viewing

# This script plots the topography obtained by
# the finest scale projection for a finite volume scheme

# change to current folder
dirname = os.path.dirname(__file__)
path    = os.path.join(dirname, "..", "out", "build", "x64-Release", "test", "results")
 
os.chdir(path)

# flow + topo files
h_file  = "depths.csv"
qx_file = "discharge_x.csv"
qy_file = "discharge_y.csv"
z_file  = "topo.csv"

# finest resolution mesh info
mesh_info_file = "mesh_info.csv"

mesh_info = pd.read_csv( os.path.join(path, mesh_info_file) )

# mesh information for use in meshgrid
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

X, Y = np.meshgrid(x, y)

h  = pd.read_csv( os.path.join(path, h_file) ) ["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
qx = pd.read_csv( os.path.join(path, qx_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
qy = pd.read_csv( os.path.join(path, qy_file) )["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]
z  = pd.read_csv( os.path.join(path, z_file) ) ["results"].values.reshape(mesh_dim, mesh_dim)[0:ysz, 0:xsz]

params = {
"legend.fontsize" : "xx-large",
"axes.labelsize"  : "large",
"axes.titlesize"  : "large",
"xtick.labelsize" : "large",
"ytick.labelsize" : "large"
}

pylab.rcParams.update(params)

sp_kw = {"projection": "3d"}

fig, ax = plt.subplots(subplot_kw=sp_kw)

ax.plot_surface(X, Y, qx)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("$q_x \: (m^s/s)$")

#plt.show()
plt.savefig("surf-qx", bbox_inches="tight")

fig, ax = plt.subplots(subplot_kw=sp_kw)

ax.plot_surface(X, Y, qy)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("$q_y \: (m^s/s)$")

#plt.show()
plt.savefig("surf-qy", bbox_inches="tight")

fig, ax = plt.subplots(subplot_kw=sp_kw)

eta = h + z

ax.plot_surface( X, Y, eta)
ax.plot_surface( X, Y, z)
#ax.set_zlim(0, 1)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_zlabel("$\eta \: (m)$")

#plt.show()
plt.savefig("surf-eta", bbox_inches="tight")

'''
fig, ax = plt.subplots(subplot_kw=sp_kw)

ax.plot_surface(X, Y, z)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

plt.show()
'''
fig, ax = plt.subplots()

ax.contourf(X, Y, z)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

plt.savefig("cont-z", bbox_inches="tight")

fig, ax = plt.subplots()

# "cs" stands for ContourSet, returned by contourf
contourset = ax.contourf(X, Y, qx)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

colorbar = fig.colorbar(contourset)
colorbar.ax.set_ylabel("$q_x \: (m^s/s)$")

plt.savefig("cont-qx", bbox_inches="tight")

fig, ax = plt.subplots()

contourset = ax.contourf(X, Y, qy)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

colorbar = fig.colorbar(contourset)
colorbar.ax.set_ylabel("$q_y \: (m^s/s)$")

plt.savefig("cont-qy", bbox_inches="tight")

fig, ax = plt.subplots()

contourset = ax.contourf(X, Y, h)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")

colorbar = fig.colorbar(contourset)
colorbar.ax.set_ylabel("$h \: (m)$")

plt.savefig("cont-h", bbox_inches="tight")

plt.clf()