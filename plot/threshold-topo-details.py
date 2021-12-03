import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
import gc

from mpl_toolkits.mplot3d import Axes3D # import for 3D viewing

def dilate(coord):
    coord &= 0x0000ffff;                         # in binary: ---- ---- ---- ---- fedc ba98 7654 3210

    coord = (coord ^ (coord << 8)) & 0x00ff00ff; # in binary: ---- ---- fedc ba98 ---- ---- 7654 3210
    coord = (coord ^ (coord << 4)) & 0x0f0f0f0f; # in binary: ---- fedc ---- ba98 ---- 7654 ---- 3210
    coord = (coord ^ (coord << 2)) & 0x33333333; # in binary: --fe --dc --ba --98 --76 --54 --32 --10
    coord = (coord ^ (coord << 1)) & 0x55555555; # in binary: -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    
    return coord;

def gen_morton_code(x, y):
    return dilate(x) | (dilate(y) << 1)

def get_level_idx(level):
    return int( ( ( 1 << (2 * level) ) - 1 ) / 3 )

def plot_level(codes, indices, data_z_order, level, file_name):
    # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
    sorted_codes, rev_z_order = [ list(tuple_) for tuple_ in zip( *sorted( zip(codes, indices) ) ) ]
    
    indices, data_row_major   = [ list(tuple_) for tuple_ in zip( *sorted( zip(rev_z_order, data_z_order) ) ) ]
    
    grid_dim = 1 << level
    
    x = [x_ for x_ in range(grid_dim)]
    y = [y_ for y_ in range(grid_dim)]
    
    X, Y = np.meshgrid(x, y)
    
    sp_kw = {"projection": "3d"}

    fig, ax = plt.subplots(subplot_kw=sp_kw)
    
    ax.scatter( X, Y, np.asarray(data_row_major).reshape(grid_dim, grid_dim), s=0.5 )
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_ylim(0, grid_dim)
    ax.set_xlim(0, grid_dim)
    #ax.set_zlim(0, 1)
    #ax.set_aspect(1)
    
    #plt.show()
    
    plt.savefig(file_name + "-level-" + str(level), bbox_inches="tight")
    
    plt.cla()
    plt.clf()
    plt.close(fig)

path = r"C:\Users\cip19aac\Google Drive\Alovya_2021\code\HFV1_GPU_2D\HFV1_GPU_2D\results\mw-debug"
os.chdir(path)

plt.rcParams.update({'font.size': 12}) 

z0_x_a_file  = "monaiz0-y-a" 
z0_x_b_file  = "monaiz0-y-b"
z0_x_g_file  = "monaiz0-y-g"
z1x_x_a_file = "monaiz1x-y-a"
z1x_x_b_file = "monaiz1x-y-b"
z1x_x_g_file = "monaiz1x-y-g"
z1y_x_a_file = "monaiz1y-y-a"
z1y_x_b_file = "monaiz1y-y-b"
z1y_x_g_file = "monaiz1y-y-g"

z0_x_a_hier  = pd.read_csv( os.path.join(path,  z0_x_a_file + ".csv") )["results"]
z0_x_b_hier  = pd.read_csv( os.path.join(path,  z0_x_b_file + ".csv") )["results"]
z0_x_g_hier  = pd.read_csv( os.path.join(path,  z0_x_g_file + ".csv") )["results"]
z1x_x_a_hier = pd.read_csv( os.path.join(path, z1x_x_a_file + ".csv") )["results"]
z1x_x_b_hier = pd.read_csv( os.path.join(path, z1x_x_b_file + ".csv") )["results"]
z1x_x_g_hier = pd.read_csv( os.path.join(path, z1x_x_g_file + ".csv") )["results"]
z1y_x_a_hier = pd.read_csv( os.path.join(path, z1y_x_a_file + ".csv") )["results"]
z1y_x_b_hier = pd.read_csv( os.path.join(path, z1y_x_b_file + ".csv") )["results"]
z1y_x_g_hier = pd.read_csv( os.path.join(path, z1y_x_g_file + ".csv") )["results"]


z0_y_a_file  = "z0-y-a" 
z0_y_b_file  = "z0-y-b"
z0_y_g_file  = "z0-y-g"
z1x_y_a_file = "z1x-y-a"
z1x_y_b_file = "z1x-y-b"
z1x_y_g_file = "z1x-y-g"
z1y_y_a_file = "z1y-y-a"
z1y_y_b_file = "z1y-y-b"
z1y_y_g_file = "z1y-y-g"

z0_y_a_hier  = pd.read_csv( os.path.join(path,  z0_y_a_file + ".csv") )["results"]
z0_y_b_hier  = pd.read_csv( os.path.join(path,  z0_y_b_file + ".csv") )["results"]
z0_y_g_hier  = pd.read_csv( os.path.join(path,  z0_y_g_file + ".csv") )["results"]
z1x_y_a_hier = pd.read_csv( os.path.join(path, z1x_y_a_file + ".csv") )["results"]
z1x_y_b_hier = pd.read_csv( os.path.join(path, z1x_y_b_file + ".csv") )["results"]
z1x_y_g_hier = pd.read_csv( os.path.join(path, z1x_y_g_file + ".csv") )["results"]
z1y_y_a_hier = pd.read_csv( os.path.join(path, z1y_y_a_file + ".csv") )["results"]
z1y_y_b_hier = pd.read_csv( os.path.join(path, z1y_y_b_file + ".csv") )["results"]
z1y_y_g_hier = pd.read_csv( os.path.join(path, z1y_y_g_file + ".csv") )["results"]

L = 9

max_grid_dim = 1 << (L - 1)

indices = [max_grid_dim * y + x  for y in range(max_grid_dim) for x in range(max_grid_dim)]
codes   = [gen_morton_code(x, y) for y in range(max_grid_dim) for x in range(max_grid_dim)]

eps = 1e-3

for level in range(L - 1, 0, -1):
    eps_level = eps / ( 1 << (L - level) )

    codes_level   = codes  [ 0 : ( 1 << (2 * level) ) ]
    indices_level = indices[ 0 : ( 1 << (2 * level) ) ]
    
    start = get_level_idx(level)
    end   = get_level_idx(level + 1)
    
    z0_x_a_z  =  z0_x_a_hier[start : end + 1]
    z0_x_b_z  =  z0_x_b_hier[start : end + 1]
    z0_x_g_z  =  z0_x_g_hier[start : end + 1]
    z1x_x_a_z = z1x_x_a_hier[start : end + 1]
    z1x_x_b_z = z1x_x_b_hier[start : end + 1]
    z1x_x_g_z = z1x_x_g_hier[start : end + 1]
    z1y_x_a_z = z1y_x_a_hier[start : end + 1]
    z1y_x_b_z = z1y_x_b_hier[start : end + 1]
    z1y_x_g_z = z1y_x_g_hier[start : end + 1]
    
    z0_x_a_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_x_a_z]
    z0_x_b_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_x_b_z]
    z0_x_g_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_x_g_z]
    z1x_x_a_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_x_a_z]
    z1x_x_b_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_x_b_z]
    z1x_x_g_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_x_g_z]
    z1y_x_a_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_x_a_z]
    z1y_x_b_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_x_b_z]
    z1y_x_g_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_x_g_z]
    
    z0_y_a_z  =  z0_y_a_hier[start : end + 1]
    z0_y_b_z  =  z0_y_b_hier[start : end + 1]
    z0_y_g_z  =  z0_y_g_hier[start : end + 1]
    z1x_y_a_z = z1x_y_a_hier[start : end + 1]
    z1x_y_b_z = z1x_y_b_hier[start : end + 1]
    z1x_y_g_z = z1x_y_g_hier[start : end + 1]
    z1y_y_a_z = z1y_y_a_hier[start : end + 1]
    z1y_y_b_z = z1y_y_b_hier[start : end + 1]
    z1y_y_g_z = z1y_y_g_hier[start : end + 1]
    
    z0_y_a_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_y_a_z]
    z0_y_b_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_y_b_z]
    z0_y_g_thresh  = [1 if abs(z) >= eps_level else 0 for z in  z0_y_g_z]
    z1x_y_a_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_y_a_z]
    z1x_y_b_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_y_b_z]
    z1x_y_g_thresh = [1 if abs(z) >= eps_level else 0 for z in z1x_y_g_z]
    z1y_y_a_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_y_a_z]
    z1y_y_b_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_y_b_z]
    z1y_y_g_thresh = [1 if abs(z) >= eps_level else 0 for z in z1y_y_g_z]
    
    z0_a_diff  = [ (x - y) for x, y in zip(z0_x_a_z , z0_y_a_z ) ]
    z0_b_diff  = [ (x - y) for x, y in zip(z0_x_b_z , z0_y_b_z ) ]
    z0_g_diff  = [ (x - y) for x, y in zip(z0_x_g_z , z0_y_g_z ) ]
    z1x_a_diff = [ (x - y) for x, y in zip(z1x_x_a_z, z1x_y_a_z) ]
    z1x_b_diff = [ (x - y) for x, y in zip(z1x_x_b_z, z1x_y_b_z) ]
    z1x_g_diff = [ (x - y) for x, y in zip(z1x_x_g_z, z1x_y_g_z) ]
    z1y_b_diff = [ (x - y) for x, y in zip(z1y_x_b_z, z1y_y_b_z) ]
    z1y_a_diff = [ (x - y) for x, y in zip(z1y_x_a_z, z1y_y_a_z) ]
    z1y_g_diff = [ (x - y) for x, y in zip(z1y_x_g_z, z1y_y_g_z) ]
    
    plot_level(codes_level, indices_level, z0_a_diff , level,  z0_x_a_file)
    plot_level(codes_level, indices_level, z0_b_diff , level,  z0_x_b_file)
    plot_level(codes_level, indices_level, z0_g_diff , level,  z0_x_g_file)
    plot_level(codes_level, indices_level, z1x_a_diff, level, z1x_x_a_file)
    plot_level(codes_level, indices_level, z1x_b_diff, level, z1x_x_b_file)
    plot_level(codes_level, indices_level, z1x_g_diff, level, z1x_x_g_file)
    plot_level(codes_level, indices_level, z1y_b_diff, level, z1y_x_a_file)
    plot_level(codes_level, indices_level, z1y_a_diff, level, z1y_x_b_file)
    plot_level(codes_level, indices_level, z1y_g_diff, level, z1y_x_g_file)