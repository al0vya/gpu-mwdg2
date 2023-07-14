import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def write_par_file(
    epsilon,
    solver,
    input_file
):
    with open(input_file, 'w') as fp:
        params = (
            "max_ref_lvl   9\n" +
            "initial_tstep 1\n" +
            "dirroot       results\n" +
            "epsilon       %s\n" +
            "fpfric        0.01\n" +
            "DEMfile       monai.dem\n" +
            "startfile     monai.start\n" +
            "bcifile       monai.bci\n" +
            "bdyfile       monai.bdy\n" +
            "stagefile     monai.stage\n" +
            "saveint       22.5\n" +
            "massint       0.1\n" +
            "sim_time      22.5\n" +
            "%s\n" +
            "cumulative\n" +
            "raster_out\n" +
            "vtk\n" +
            "cuda\n" +
            "refine_wall\n" +
            "ref_thickness 16\n" +
            "wall_height   0.5"
        ) % (
            epsilon,
            solver
        )
        
        fp.write(params)

def plot_depths(
    depths_verified,
    depths_computed,
    filename
):
    exp_data = np.loadtxt(fname="experimental.txt", skiprows=1, delimiter=',')
    
    t_exp      = exp_data[:,0]
    depths_exp = exp_data[:,1]
    
    N_points_verified = len(depths_verified)
    N_points_computed = len(depths_computed)
    
    t_min = 0
    t_max = 22.5
    
    dt_verified = (t_max - t_min) / N_points_verified
    dt_computed = (t_max - t_min) / N_points_computed
    
    t_verified = [ t_min + dt_verified * i for i in range(N_points_verified) ]
    t_computed = [ t_min + dt_computed * i for i in range(N_points_computed) ]
    
    fig, ax = plt.subplots()
    
    ax.plot(t_verified, depths_verified, label="verified")
    ax.plot(t_computed, depths_computed, label="computed")
    ax.plot(t_exp,      depths_exp,      label="experimental")
    
    plt.setp(ax,
        xlim=(t_min, t_max),
        xlabel=r"$t \, (s)$",
        ylabel=r"$h + z \, (m)$"
    )
    
    ax.legend()
    
    fig.savefig(fname=os.path.join("results", filename), bbox_inches="tight")
    
    plt.close()
    
def verify_depths(
    epsilon,
    solver,
    depths_computed
):
    filename = ""
    
    if epsilon == 0:
        solver_text = "fv1"   if solver == "hwfv1" else "dg2"
    else:
        solver_text = "hwfv1" if solver == "hwfv1" else "mwdg2"
    
    verification_filename = "stage-" + solver_text + ".txt"
    
    depths_verified = np.loadtxt(
        fname=verification_filename,
        skiprows=1,
        usecols=1,
        delimiter=','
    )
    
    plot_depths(
        depths_verified=depths_verified,
        depths_computed=depths_computed,
        filename=solver_text
    )
    
    error = np.abs(depths_computed - depths_verified).mean()
    
    return "passed" if (error < 1e-14) else "failed"

def verify(
    epsilon,
    solver
):
    print( "Running simulation, solver: " + solver + ", eps = " + str(epsilon) )
    
    input_file = "test.par"
    
    write_par_file(
        epsilon=epsilon,
        solver=solver,
        input_file=input_file
    )
    
    subprocess.run( [os.path.join("..", "gpu-mwdg2.exe"), input_file] )
    
    depths_computed = np.loadtxt(fname=os.path.join("results", "res.stage"), skiprows=7, usecols=1, delimiter=' ')
    
    return verify_depths(epsilon, solver, depths_computed)
    
def verify_all():
    verification = {}
    
    verification["hwfv1"]       = {}
    verification["hwfv1"][0]    = {}
    verification["hwfv1"][1e-3] = {}
    
    verification["mwdg2"]       = {}
    verification["mwdg2"][0]    = {}
    verification["mwdg2"][1e-3] = {}
    
    verification["hwfv1"][0]    = verify(epsilon=0,    solver="hwfv1")
    verification["hwfv1"][1e-3] = verify(epsilon=1e-3, solver="hwfv1")
    verification["mwdg2"][0]    = verify(epsilon=0,    solver="mwdg2")
    verification["mwdg2"][1e-3] = verify(epsilon=1e-3, solver="mwdg2")
    
    return verification
    
def EXIT_HELP():
    help_message = ("Use this tool as:\n" + "python test.py <OPTION>, OPTION={fast|slow} to perform either fast or slow verification.\n")
    
    sys.exit(help_message)

def main():
    if len(sys.argv) != 2:
        EXIT_HELP()
    
    dummy, option = sys.argv
    
    print("Running verification test...")
    
    monai_dir = os.path.join("..", "monai")
    
    subprocess.run( [ "python", os.path.join(monai_dir, "stage.py" ) ] )
    subprocess.run( [ "python", os.path.join(monai_dir, "inflow.py") ] )
    subprocess.run( [ "python", os.path.join(monai_dir, "raster.py") ] )
    
    print(option)
    
    if option == "fast":
        print("Code " + verify(epsilon=1e-3, solver="hwfv1") + " fast verification.\n")
    elif option == "slow":
        verification = verify_all()
        
        results = (
            "%8s" + "%8s" + "%8s\n\n" +
            "%8s" + "%8s" + "%8s\n\n" +
            "%8s" + "%8s" + "%8s"
        ) % (
            "eps", "0", "1e-3",
            "hwfv1", str( verification["hwfv1"][0] ), str( verification["hwfv1"][1e-3] ),
            "mwdg2", str( verification["mwdg2"][0] ), str( verification["mwdg2"][1e-3] )
        )
        
        print(results)
    else:
        EXIT_HELP()
        
if __name__ == "__main__":
    main()