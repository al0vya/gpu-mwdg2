import os
import sys
import shutil
import subprocess

def main():
    generate_input_files()
    
    executable = "gpu-mwdg2.exe" if sys.platform == "win32" else "gpu-mwdg2"
    
    solvers = ["hwfv1", "mwdg2"]
    
    input_files = ["unit_tests_HW.par","unit_tests_MW.par"]
    
    for solver, input_file in zip(solvers, input_files):
        print(f"Running {solver} simulation...")
        
        write_par_file(solver, input_file)
        
        subprocess.run( [os.path.join("..", executable), input_file] )
        
    for input_file in input_files:
        shutil.copy( input_file, os.path.join("res", input_file) )
        
    write_dummy_dem_file()

def generate_input_files():
    print("Generating input files...")
    
    subprocess.run( [ "python", "stage.py"  ] )
    subprocess.run( [ "python", "inflow.py" ] )
    subprocess.run( [ "python", "raster.py" ] )
    
def write_par_file(
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
            "epsilon       1e-3\n" +
            "wall_height   0.5\n" +
            "initial_tstep 1\n" +
            "fpfric        0.01\n" +
            "sim_time      0.1\n" +
            "massint       0.1\n" +
            "saveint       0.1\n" +
            "DEMfile       monai.dem\n" +
            "startfile     monai.start\n" +
            "bcifile       monai.bci\n" +
            "bdyfile       monai.bdy\n" +
            "stagefile     monai.stage\n"
        )
        
        fp.write(params)

def write_dummy_dem_file():
    dummy_dem_file = (
        "ncols        392\n" +
        "nrows        243\n" +
        "xllcorner    0\n" +
        "yllcorner    0\n" +
        "cellsize     0.014\n" +
        "NODATA_value -9999\n"
    )
    
    with open(os.path.join("res", "monai.txt"), 'w') as fp:
        fp.write(dummy_dem_file)
        
if __name__ == "__main__":
    main()