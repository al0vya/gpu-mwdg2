import subprocess
import sys
import os

path       = os.path.join("..", "out", "build", "x64-Release")
input_file = os.path.join(path, "test", "inputs.par")
solver     = os.path.join(path, "gpu-mwdg2.exe")
results    = os.path.join(path, "test", "results")

print(results)

params = r"""
test_case 	4
max_ref_lvl	7
min_dt		1
respath		%s
epsilon		0
fpfric 		0.01
tol_h		1e-4
tol_q		0
tol_s		1e-9
g			9.80665
CFL			0.33
massint		0.05
solver		mw
wall_height	0
""" % results

with open(input_file, 'w') as fp:
    fp.write(params)

subprocess.run( [solver, input_file] )

import plot_2D_solution