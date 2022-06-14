#pragma once

#include "SolverParams.h"
#include "AssembledSolution.h"

#include "write_cumulative_data.h"

#include "append_reals_to_file.cuh"

__host__
void write_c_prop_data
(
	const char*              respath,
	const clock_t&           start,
	const SolverParams&      solver_params,
	const SimulationParams&  sim_params,
	const AssembledSolution& d_assem_sol,
	const real&              time_now,
	const real&              dt,
	const int&               num_cells,
	const bool&              first_t_step
);