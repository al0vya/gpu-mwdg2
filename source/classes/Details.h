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

	Details
	(
		const int& num_details, 
		const int& solver_type
	)
	:
		eta0(num_details),
		qx0 (num_details),
		qy0 (num_details),
		z0  (num_details),

		eta1x( (solver_type == MWDG2) ? num_details : 0 ),
		qx1x ( (solver_type == MWDG2) ? num_details : 0 ),
		qy1x ( (solver_type == MWDG2) ? num_details : 0 ),
		z1x  ( (solver_type == MWDG2) ? num_details : 0 ),
		
		eta1y( (solver_type == MWDG2) ? num_details : 0 ),
		qx1y ( (solver_type == MWDG2) ? num_details : 0 ),
		qy1y ( (solver_type == MWDG2) ? num_details : 0 ),
		z1y  ( (solver_type == MWDG2) ? num_details : 0 )
	{}

} Details;