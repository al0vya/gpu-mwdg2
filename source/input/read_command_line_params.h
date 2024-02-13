#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"
#include "../classes/PlottingParams.h"

void read_command_line_params
(
	const int&        argc, 
	char**            argv,
	SimulationParams& sim_params,
	SolverParams&     solver_params,
	PlottingParams&   plot_params
);