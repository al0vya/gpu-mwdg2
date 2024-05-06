#pragma once

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include "../classes/SimulationParams.h"
#include "../classes/PlottingParams.h"

void write_cumulative_data
(
	const real&             current_time,
	const real&             time_mra,
	const real&             time_solver,
	const real&             dt,
	const int&              num_cells,
	const SimulationParams& sim_params,
	const PlottingParams&   plot_params,
	const bool              first_t_step
);