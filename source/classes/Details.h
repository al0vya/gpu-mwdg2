#pragma once

#include <algorithm>

#include "SubDetails.h"
#include "../types/SolverTypes.h"

class Details
{
public:
	Details
	(
		const SolverParams& solver_params
	);

	Details
	(
		const SolverParams& solver_params,
		const char*         dirroot,
		const char*         prefix
	);

	void write_to_file
	(
		const char* dirroot,
		const char* prefix
	);

	real verify
	(
		const char* dirroot,
		const char* prefix
	);

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
};