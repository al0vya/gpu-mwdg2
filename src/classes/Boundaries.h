#pragma once

#include "Boundary.h"

typedef struct Boundaries
{
	Boundary north;
	Boundary east;
	Boundary south;
	Boundary west;

	Boundaries
	(
		const char*                 input_filename,
		const SimulationParams& sim_params,
		const real&                 cell_size,
		const int&                  test_case
	)
	:
		north(input_filename, sim_params, cell_size, test_case, NORTH),
		east (input_filename, sim_params, cell_size, test_case, EAST),
		south(input_filename, sim_params, cell_size, test_case, SOUTH),
		west (input_filename, sim_params, cell_size, test_case, WEST)
	{
		if (test_case != 0) fprintf(stdout, "Running built-in test case, using open boundary conditions.\n");
	}

	void update_all_inlets
	(
		const char* input_filename,
		const real& time_now
	)
	{
		north.update_inlet(time_now);
		east.update_inlet (time_now);
		south.update_inlet(time_now);
		west.update_inlet (time_now);
	}

} Boundaries;