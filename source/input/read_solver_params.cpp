#include "read_solver_params.h"

SolverParams read_solver_params
(
	const char* input_filename
)
{
	SolverParams solver_params = SolverParams();

	solver_params.L           = read_keyword_int (input_filename, "max_ref_lvl", 11);
	solver_params.min_dt      = read_keyword_real(input_filename, "min_dt", 6);
	solver_params.epsilon     = read_keyword_real(input_filename, "epsilon", 7);
	solver_params.tol_h       = read_keyword_real(input_filename, "tol_h", 5);
	solver_params.tol_q       = read_keyword_real(input_filename, "tol_q", 5);
	solver_params.tol_s       = read_keyword_real(input_filename, "tol_s", 5);
	solver_params.wall_height = read_keyword_real(input_filename, "wall_height", 11);

	char solvertype_buf[128] = {'\0'};
	read_keyword_str(input_filename, "solver", 6, solvertype_buf);

	if ( !strncmp(solvertype_buf, "hw", 2) )
	{
		solver_params.solver_type = HWFV1;
		solver_params.CFL         = C(0.5);
	}
	else if ( !strncmp(solvertype_buf, "mw", 2) )
	{
		solver_params.solver_type = MWDG2;
		solver_params.CFL         = C(0.3);
	}
	else
	{
		fprintf(stderr, "Error: invalid solver type specified, please specify either \"hw\" or \"mw\", file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

    read_keyword_str(input_filename, "grading", 7, solvertype_buf);
    solver_params.grading = ( !strncmp(solvertype_buf, "on", 2) );
	
    read_keyword_str(input_filename, "limitslopes", 11, solvertype_buf);
    solver_params.limitslopes = ( !strncmp(solvertype_buf, "on", 2) );

	if (solver_params.limitslopes)
	{
		solver_params.tol_Krivo = read_keyword_real(input_filename, "tol_Krivo", 9);
	}
	
    read_keyword_str(input_filename, "refine_wall", 11, solvertype_buf);
    solver_params.refine_wall = ( !strncmp(solvertype_buf, "on", 2) );

	if (solver_params.refine_wall)
	{
		solver_params.ref_thickness = read_keyword_int(input_filename, "ref_thickness", 13);
	}

	return solver_params;
}