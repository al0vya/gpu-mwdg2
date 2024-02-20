#pragma once

#include "SubDetails.h"
#include "../types/SolverTypes.h"

typedef struct Details
{
	SubDetails eta0;
	SubDetails qx0;
	SubDetails qy0;
	SubDetails z0;

	SubDetails eta1x;
	SubDetails qx1x;
	SubDetails qy1x;
	SubDetails z1x;

	SubDetails eta1y;
	SubDetails qx1y;
	SubDetails qy1y;
	SubDetails z1y;

	int solver_type = 0;

	Details
	(
		const SolverParams& solver_params
	)
	:
		eta0(solver_params),
		qx0 (solver_params),
		qy0 (solver_params),
		z0  (solver_params),

		eta1x( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qx1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qy1x ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		z1x  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		
		eta1y( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qx1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		qy1y ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),
		z1y  ( (solver_params.solver_type == MWDG2) ? solver_params : SolverParams() ),

		solver_type(solver_params.solver_type)
	{}

} Details;