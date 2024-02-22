import os
import sys
import collections
import subprocess
import numpy as np
import matplotlib.pyplot as plt

def EXIT_HELP():
    help_message = ("Use this tool as:\n" + "python test.py <OPTION>, OPTION={fast|slow} to perform either fast or slow verification.\n")
    
    sys.exit(help_message)
    
def main():
    if len(sys.argv) != 2:
        EXIT_HELP()
    
    dummy, option = sys.argv
    
    generate_input_files()
    
    print("Running verification test...")
    
    if option == "fast":
        error = compute_error(epsilon=1e-3, solver="hwfv1")
        
        print(f"Mean absolute error for hwfv1 solver, epsilon = 1e-3: {error}")
    elif option == "slow":
        errors = compute_all_errors()
        
        print(f"Mean absolute error for hwfv1 solver, epsilon = 1e-3: {errors['hwfv1'][1e-3]}")
        print(f"Mean absolute error for hwfv1 solver, epsilon = 0:    {errors['hwfv1'][0]}")
        print(f"Mean absolute error for mwdg2 solver, epsilon = 1e-3: {errors['mwdg2'][1e-3]}")
        print(f"Mean absolute error for mwdg2 solver, epsilon = 0:    {errors['mwdg2'][0]}")
    else:
        EXIT_HELP()
        
def write_par_file(
    epsilon,
    solver,
    input_file
):
    with open(input_file, 'w') as fp:
        params = (
            "monai\n" +
            f"{solver}\n" +
            "cuda\n" +
            "cumulative\n" +
            "refine_wall\n" +
            "ref_thickness 16\n" +
            "max_ref_lvl   9\n" +
            f"epsilon       {epsilon}\n" +
            "wall_height   0.5\n" +
            "initial_tstep 1\n" +
            "fpfric        0.01\n" +
            "sim_time      22.5\n" +
            "massint       0.1\n" +
            "saveint       22.5\n" +
            "DEMfile       monai.dem\n" +
            "startfile     monai.start\n" +
            "bcifile       monai.bci\n" +
            "bdyfile       monai.bdy\n" +
            "stagefile     monai.stage\n"
        )
        
        fp.write(params)

def plot_depths(
    depths_verified,
    depths_computed,
    filename
):
    exp_data = np.loadtxt(fname="MonaiValley_WaveGages.txt", skiprows=1, delimiter='\t')
    
    t_exp      = exp_data[:,0]
    depths_exp = exp_data[:,1] / 100 # convert from cm to m
    
    N_points_verified = len(depths_verified)
    N_points_computed = len(depths_computed)
    
    t_min = 0
    t_max = 22.5
    
    dt_verified = (t_max - t_min) / N_points_verified
    dt_computed = (t_max - t_min) / N_points_computed
    
    t_verified = [ t_min + dt_verified * i for i in range(N_points_verified) ]
    t_computed = [ t_min + dt_computed * i for i in range(N_points_computed) ]
    
    fig, ax = plt.subplots()
    
    ax.plot(t_verified, depths_verified, label="Verified")
    ax.plot(t_computed, depths_computed, label="Computed")
    ax.plot(t_exp,      depths_exp,      label="Experimental")
    
    plt.setp(ax,
        xlim=(t_min, t_max),
        xlabel="$t$ (s)",
        ylabel="$h + z$ (m)"
    )
    
    ax.legend()
    
    fig.savefig(fname=os.path.join("res", filename) + ".png", bbox_inches="tight")
    
    plt.close()
    
def compute_error(
    epsilon,
    solver
):
    print(f"Running simulation, solver: {solver}, eps: {epsilon}")
    
    input_file = "functional-test.par"
    
    write_par_file(
        epsilon=epsilon,
        solver=solver,
        input_file=input_file
    )
    
    executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
    
    subprocess.run( [os.path.join("..", executable), input_file] )
    
    solver_text = f"{solver}-eps-{epsilon}"
    
    depths_verified = np.loadtxt(
        fname=os.path.join(solver_text, "res.txt"),
        skiprows=9,
        usecols=2,
        delimiter=' '
    )
    
    depths_computed = np.loadtxt(fname=os.path.join("res", "res.stage"),
        skiprows=9,
        usecols=2,
        delimiter=' '
    )
    
    plot_depths(
        depths_verified=depths_verified,
        depths_computed=depths_computed,
        filename=solver_text
    )
    
    error = np.abs(depths_computed - depths_verified).mean()
    
    return error
    
def compute_all_errors():
    rec_dd = lambda: collections.defaultdict(rec_dd)
    
    errors = rec_dd()
    
    errors["hwfv1"][0]    = compute_error(epsilon=0,    solver="hwfv1")
    errors["hwfv1"][1e-3] = compute_error(epsilon=1e-3, solver="hwfv1")
    errors["mwdg2"][0]    = compute_error(epsilon=0,    solver="mwdg2")
    errors["mwdg2"][1e-3] = compute_error(epsilon=1e-3, solver="mwdg2")
    
    return errors
    
def generate_input_files():
    print("Generating input files...")
    
    subprocess.run( [ "python", "stage.py"  ] )
    subprocess.run( [ "python", "inflow.py" ] )
    subprocess.run( [ "python", "raster.py" ] )

if __name__ == "__main__":
    main()