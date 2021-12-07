# script to plot hierarchy of scale coefficients or details
# to run, do python plot-z-order-data.py <FILE_NAME> <REFINEMENT_LEVEL>

import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys

from mpl_toolkits.mplot3d import Axes3D # import for 3D viewing

path = r"C:\Users\alovy\Documents\HFV1_GPU_2D\HFV1_GPU_2D\results"
os.chdir(path)

plt.rcParams.update({'font.size': 12}) 

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
    
    plt.show()
    
    #plt.savefig(file_name + "-level-" + str(level), bbox_inches="tight")

file_name = sys.argv[1]

L_str = sys.argv[2]

L = int(L_str)

data_hierarchy = pd.read_csv( os.path.join(path, file_name + ".csv") )["results"]

max_grid_dim = 1 << L

indices = [max_grid_dim * y + x  for y in range(max_grid_dim) for x in range(max_grid_dim)]
codes   = [gen_morton_code(x, y) for y in range(max_grid_dim) for x in range(max_grid_dim)]

for level in range(L, -1, -1):
    show_next = input("Show next level (y/n)? ")
    
    if (show_next == "y"):
        codes_level   = codes  [ 0 : ( 1 << (2 * level) ) ]
        indices_level = indices[ 0 : ( 1 << (2 * level) ) ]
        
        start = get_level_idx(level)
        end   = get_level_idx(level + 1)
        
        data_z_order = data_hierarchy[start : end + 1]
        
        plot_level(codes_level, indices_level, data_z_order, level, file_name)
    else:
        quit("Exiting")

'''
python plot-z-order-data.py details-z0-a 7
python plot-z-order-data.py details-z0-b 7
python plot-z-order-data.py details-z0-g 7
'''