import subprocess
import sys
import os
from plot_2D_solution import Solution

def set_params(test_case, results, epsilon, massint, solver, input_file):
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
    "wall_height	0") % (test_case, results, epsilon, massint, solver)

    with open(input_file, 'w') as fp:
        fp.write(params)

path        = os.path.join("..", "out", "build", "x64-Release")
input_file  = os.path.join(path, "test", "inputs.par")
solver_file = os.path.join(path, "gpu-mwdg2.exe")
results     = os.path.join(path, "test", "results")

num_tests = 18

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

for test in range(2, 4):
    test_case = test + 1
    epsilon   = 1e-3
    massint   = massints[test]
    test_name = test_names[test]
    solver    = "mw"

    set_params(test_case, results, epsilon, massint, solver, input_file)

    subprocess.run( [solver_file, input_file] )
    
    Solution().plot_soln(test_case, test_name)

