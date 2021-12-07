import subprocess

import Solution
import DischargeErrors

def set_params(test_case, 
    results, 
    epsilon, 
    massint, 
    solver, 
    row_major, 
    c_prop, 
    vtk, 
    input_file
):
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
    "c_prop %s\n" +
    "vtk        %s") % (test_case, results, epsilon, massint, solver, row_major, c_prop, vtk)

    with open(input_file, 'w') as fp:
        fp.write(params)

class Test:
    def __init__(self,
        test_case, 
        eps, 
        massint, 
        test_name, 
        solver, 
        c_prop_tests, 
        results, 
        input_file
    ):
        self.test_case = test_case
        self.epsilon   = epsilon
        self.massint   = massint 
        self.test_name = test_name
        self.solver    = solver

        if self.test_case in c_prop_tests:
            self.row_major = "off"
            self.vtk       = "off"
            self.c_prop    = "on"
        else:
            self.row_major = "on"
            self.vtk       = "on"
            self.c_prop    = "off"

        self.results    = results
        self.input_file = input_file

    def set_params(self):
        set_params(
            self.test_case,
            self.results,
            self.epsilon, 
            self.massint, 
            self.solver, 
            self.row_major, 
            self.c_prop, 
            self.vtk,
            self.input_file
        )

    def run_test(self, solver_file):
        self.set_params()

        subprocess.run( [solver_file, self.input_file] )

        if self.c_prop == "on":
            DischargeErrors(self.solver).plot_errors(self.test_case, self.test_name)
        else:
            Solution().plot_soln(self.test_case, self.test_name)