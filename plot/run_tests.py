import subprocess
import sys
import os
from plot_2D_solution import Solution
from plot_c_prop      import DischargeErrors

def set_params(test_case, results, epsilon, massint, solver, row_major, c_prop, input_file):
    params = ("" +
    "test_case   %s\n" +
    "max_ref_lvl	7\n" +
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
    "c_prop %s") % (test_case, results, epsilon, massint, solver, row_major, c_prop)

    with open(input_file, 'w') as fp:
        fp.write(params)

path        = os.path.join("..", "out", "build", "x64-Release")
input_file  = os.path.join(path, "test", "inputs.par")
solver_file = os.path.join(path, "gpu-mwdg2.exe")
results     = os.path.join(path, "test", "results")

test_names = [
"1D-c-prop-x-dir",
"1D-c-prop-y-dir",
"1D-c-prop-x-dir",
"1D-c-prop-y-dir",
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
    test_case = test + 1
    epsilon   = 0
    massint   = massints[test]
    test_name = test_names[test]
    row_major = "off" if test_case in c_prop_tests else "on"
    c_prop    = "on"  if test_case in c_prop_tests else "off"
    solver    = "mw"

    set_params(test_case, results, epsilon, massint, solver, row_major, c_prop, input_file)

    if test_case not in c_prop_tests:
        subprocess.run( [solver_file, input_file] ) 
    
    if test_case in c_prop_tests:
        a = 1
        #DischargeErrors(solver).plot_errors(test_case, solver, test_name)
    else:
        Solution().plot_soln(test_case, test_name)