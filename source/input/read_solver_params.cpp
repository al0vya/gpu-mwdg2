#include "read_solver_params.h"

SolverParams read_solver_params
(
	const char* input_filename
)
{
	SolverParams solver_params = SolverParams();

	solver_params.L             = read_keyword_int (input_filename, "max_ref_lvl", 11);
	solver_params.initial_tstep = read_keyword_real(input_filename, "initial_tstep", 13);
	solver_params.epsilon       = read_keyword_real(input_filename, "epsilon", 7);
	solver_params.wall_height   = read_keyword_real(input_filename, "wall_height", 11);

	if ( read_keyword_bool(input_filename, "hwfv1", 5) )
	{
		solver_params.solver_type = HWFV1;
		solver_params.CFL         = C(0.5);
	}
	else if ( read_keyword_bool(input_filename, "mwdg2", 5) )
	{
		solver_params.solver_type = MWDG2;
		solver_params.CFL         = C(0.3);
	}
	else
	{
		fprintf(stderr, "Error: invalid adaptive solver type specified, please specify either \"hwfv1\" or \"mwdg2\", file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	solver_params.grading = read_keyword_bool(input_filename, "grading", 7);
    
	solver_params.limitslopes = read_keyword_bool(input_filename, "limitslopes", 11);
    
	if (solver_params.limitslopes)
	{
		solver_params.tol_Krivo = read_keyword_real(input_filename, "tol_Krivo", 9);
	}
	
	solver_params.refine_wall = read_keyword_bool(input_filename, "refine_wall", 11);
    
	if (solver_params.refine_wall)
	{
		solver_params.ref_thickness = read_keyword_int(input_filename, "ref_thickness", 13);
	}

	solver_params.startq2d = read_keyword_bool(input_filename, "startq2d", 8);

	return solver_params;
}