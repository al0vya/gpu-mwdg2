#pragma once

#include "real.h"
#include "SolverTypes.h"

typedef struct SolverParams
{
	int  L;
	real min_dt;
	real epsilon;
	real tol_h;
	real tol_q;
	real tol_s;
	real wall_height;
	int  solver_type;
	real CFL;

} SolverParams;