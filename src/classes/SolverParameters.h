#pragma once

#include "real.h"
#include "SolverTypes.h"

typedef struct SolverParameters
{
	real CFL;
	real min_dt;
	real tol_h;
	real tol_q;
	real tol_s;
	real epsilon;
	real wall_height;
	int  L;
	int  solver_type;

} SolverParameters;