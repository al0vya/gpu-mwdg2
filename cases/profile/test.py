import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def write_par_file(
    epsilon,
    solver
):
    with open("profile.par", 'w') as fp:
        params = (
            "test_case   0\n" +
            "max_ref_lvl 9\n" +
            "min_dt      1\n" +
            "respath     results\n" +
            "epsilon     %s\n" +
            "fpfric      0.01\n" +
            "rasterroot  monai\n" +
            "bcifile     monai.bci\n" +
            "bdyfile     monai.bdy\n" +
            "stagefile   monai.stage\n" +
            "tol_h       1e-3\n" +
            "tol_q       0\n" +
            "tol_s       1e-9\n" +
            "g           9.80665\n" +
            "massint     0.1\n" +
            "sim_time    22.5\n" +
            "solver      %s\n" +
            "cumulative  on\n" +
            "wall_height 0.5"
        ) % (epsilon, solver)
        
        fp.write(params)

def verify_depths(
    epsilon,
    solver,
    depths_computed
):
    filename = ""
    
    if epsilon == 0:
        filename = "stage-fv1.txt"   if solver == "hw" else "stage-dg2.txt"
    else:
        filename = "stage-hwfv1.txt" if solver == "hw" else "stage-mwdg2.txt"
    
    depths_verified = np.loadtxt(fname=filename, skiprows=1, usecols=1, delimiter=',')
    
    error = np.abs(depths_computed - depths_verified).mean()
    
    return "passed" if (error < 1e-14) else "failed"

def verify(
    epsilon,
    solver
):
    print( "Running simulation, solver: " + solver + ", eps = " + str(epsilon) )
    
    write_par_file(epsilon, solver)
    
    subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), "profile.par"] )
    
    depths_computed = np.loadtxt(fname=os.path.join("results", "stage.wd"), skiprows=7, usecols=1, delimiter=' ')
    
    return verify_depths(epsilon, solver, depths_computed)
    
def verify_all():
    verification = {}
    
    verification["hw"]       = {}
    verification["hw"][0]    = {}
    verification["hw"][1e-3] = {}
    
    verification["mw"]       = {}
    verification["mw"][0]    = {}
    verification["mw"][1e-3] = {}
    
    verification["hw"][0]    = verify(epsilon=0,    solver="hw")
    verification["hw"][1e-3] = verify(epsilon=1e-3, solver="hw")
    verification["mw"][0]    = verify(epsilon=0,    solver="mw")
    verification["mw"][1e-3] = verify(epsilon=1e-3, solver="mw")
    
    return verification
    
def main():
    if len(sys.argv) < 2:
        help_message = ("Use this tool as:\n" + "python test.py <OPTION>, OPTION={fast|slow} to perform either fast or slow verification.\n")
        
        sys.exit(help_message)
    
    dummy, option = sys.argv
    
    print("Running verification test...")
    
    monai_dir = os.path.join("..", "monai")
    
    subprocess.run( [ "python", os.path.join(monai_dir, "stage.py" ) ] )
    subprocess.run( [ "python", os.path.join(monai_dir, "inflow.py") ] )
    subprocess.run( [ "python", os.path.join(monai_dir, "raster.py") ] )
    
    if option == "fast":
        print("Code " + verify(epsilon=1e-3, solver="hw") + " fast verification.\n")
    elif option == "slow":
        verification = verify_all()
        
        results = (
            "%8s" + "%8s" + "%8s\n\n" +
            "%8s" + "%8s" + "%8s\n\n" +
            "%8s" + "%8s" + "%8s"
        ) % (
            "eps", "0", "1e-3",
            "hw", str( verification["hw"][0] ), str( verification["hw"][1e-3] ),
            "mw", str( verification["mw"][0] ), str( verification["mw"][1e-3] )
        )
        
        print(results)
    else:
        sys.exit(help_message)
        
if __name__ == "__main__":
    main()