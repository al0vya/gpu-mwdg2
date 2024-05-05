#pragma once

#include "../types/real.h"
#include "../types/SolverTypes.h"

#include "../input/read_keyword_int.h"
#include "../input/read_keyword_bool.h"
#include "../input/read_keyword_real.h"

class SolverParams
{
public:
	SolverParams();

	SolverParams
	(
		const char* input_filename
	);
	
	int  L             = 0;
	real initial_tstep = C(0.0);
	real epsilon       = C(0.0);
	real tol_h         = C(1e-3);
	real tol_q         = C(0.0);
	real tol_s         = C(1e-9);
	real wall_height   = C(0.0);
	int  solver_type   = 0;
	real CFL           = C(0.0);
	bool grading       = false;
	bool limitslopes   = false;
	real tol_Krivo     = C(9999.0);
	bool refine_wall   = false;
	int  ref_thickness = 0;
	bool startq2d      = false;
};