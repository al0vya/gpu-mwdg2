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
		const real& current_time
	)
	{
		north.update_inlet(current_time);
		east.update_inlet (current_time);
		south.update_inlet(current_time);
		west.update_inlet (current_time);
	}

} Boundaries;