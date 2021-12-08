import os
import sys

def EXIT_HELP():
    help_message = (
        "This tool is used to view a hierarchy of 2^n by 2^n grid arranged in Z order. Run using:\n" +
        "python zorder.py <MODE> <TEST_CASE_FOLDER> <FILENAME> <MAX_REFINEMENT_LEVEL>, MODE=[debug,release]"
    )
    
    sys.exit(help_message)

if len(sys.argv) < 5:
    EXIT_HELP()

import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt

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

def get_level_idx(
    level
):
    return int( ( ( 1 << (2 * level) ) - 1 ) / 3 )

class Hierarchy:
    def __init__(
        self,
        mode,
        testdir,
        filename,
        max_ref_lvl
    ):
        self.relativepath = ""

        if (mode == "debug"):
            self.relativepath = os.path.join("..", "out", "build", "x64-Debug", testdir, "results")
        elif (mode == "release"):
            self.relativepath = os.path.join("..", "out", "build", "x64-Release", testdir, "results")
        else:
            EXIT_HELP()
        
        self.savepath = os.path.join(os.path.dirname(__file__), self.relativepath)
        
        self.testdir     = testdir
        self.filename    = filename
        self.max_ref_lvl = max_ref_lvl
        
        self.data_z_order = pd.read_csv( os.path.join(self.savepath, self.filename) ).values.tolist()
                
        max_grid_dim = 1 << self.max_ref_lvl
        
        self.indices = [max_grid_dim * y + x  for y in range(max_grid_dim) for x in range(max_grid_dim)]
        self.codes   = [gen_morton_code(x, y) for y in range(max_grid_dim) for x in range(max_grid_dim)]
        
         
    def show_level(
        self,
        level
    ):
        codes_level   = self.codes  [ 0 : ( 1 << (2 * level) ) ]
        indices_level = self.indices[ 0 : ( 1 << (2 * level) ) ]
        
        start = get_level_idx(level)
        end   = get_level_idx(level + 1)
        
        data_z_order_level = self.data_z_order[start : end + 1]
        
        # https://stackoverflow.com/questions/9764298/how-to-sort-two-lists-which-reference-each-other-in-the-exact-same-way
        sorted_codes, rev_z_order = [ list(tuple_) for tuple_ in zip( *sorted( zip(codes_level, indices_level) ) ) ]
        
        indices, data_row_major   = [ list(tuple_) for tuple_ in zip( *sorted( zip(rev_z_order, data_z_order_level) ) ) ]
        
        grid_dim = 1 << level
        
        x = [x_ for x_ in range(grid_dim)]
        y = [y_ for y_ in range(grid_dim)]
        
        X, Y = np.meshgrid(x, y)
        
        fig, ax = plt.subplots(subplot_kw={"projection":"3d"})
        
        ax.scatter( X, Y, np.asarray(data_row_major).reshape(grid_dim, grid_dim), s=0.5 )
        ax.set_xlabel("i")
        ax.set_ylabel("j")
        ax.set_ylim(0, grid_dim)
        ax.set_xlim(0, grid_dim)
        
        plt.show()
        
    def show_all_levels(
        self
    ):
        for level in range(self.max_ref_lvl, -1, -1):
            show_next = input("Show next level (y/n)? ")
            
            if (show_next == "y"):
                self.show_level(level)
            else:
                sys.exit("Exiting")
    
plt.rcParams.update({'font.size': 12}) 

mode = sys.argv[1]

if mode == "debug" or mode == "release":
    dummy, mode, testdir, filename, max_ref_lvl = sys.argv
    
    Hierarchy( mode, testdir, filename, int(max_ref_lvl) ).show_all_levels()
