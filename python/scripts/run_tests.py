# this script runs all the in-built test cases of gpu-mwdg2

import sys
import os

path = ""

# if command line argument is given
if len(sys.argv) > 1:
    mode = sys.argv[1]

    if mode == "debug":
        path = os.path.join("..", "..", "out", "build", "x64-Debug")
    elif mode == "release":
        path = os.path.join("..", "..", "out", "build", "x64-Release")
    else:
        sys.exit("Please specify either debug or release in the command line.")
else:
    sys.exit("Please specify either \"debug\" or \"release\" in the command line.")

import matplotlib.pylab as pylab

# adding folders to sys.path for importing
# see: https://stackoverflow.com/questions/31291608/effect-of-using-sys-path-insert0-path-and-sys-pathappend-when-loading-modul
package_folders = [
    "classes"
]

for folder in package_folders:
    sys.path.insert( 1, os.path.join(os.path.dirname(__file__), "..", folder) )

from Test import Test

# from: https://stackoverflow.com/questions/12444716/how-do-i-set-the-figure-title-and-axes-labels-font-size-in-matplotlib
params = {
'legend.fontsize' : 'xx-large',
'axes.labelsize'  : 'xx-large',
'axes.titlesize'  : 'xx-large',
'xtick.labelsize' : 'xx-large',
'ytick.labelsize' : 'xx-large'
}

pylab.rcParams.update(params)
        
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
"dry-dam-break-wh-fric-x-dir",
"dry-dam-break-wh-fric-y-dir",
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
    Test(test + 1, 0, massints[test], test_names[test], "mw", c_prop_tests, results, input_file, mode).run_test(solver_file)