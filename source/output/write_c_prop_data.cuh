#pragma once

#include "../classes/SolverParams.h"
#include "../classes/PlottingParams.h"
#include "../classes/AssembledSolution.h"

#include "write_cumulative_data.h"

#include "append_reals_to_file.cuh"

__host__
void write_c_prop_data
(
	const PlottingParams&    plot_params,
	const clock_t&           start,
	const SolverParams&      solver_params,
	const SimulationParams&  sim_params,
	const AssembledSolution& d_assem_sol,
	const real&              current_time,
	const real&              inst_time_mra,
	const real&              cumu_time_mra,
	const real&              inst_time_solver,
	const real&              cumu_time_solver,
	const real&              dt,
	const int&               num_timesteps,
	const int&               num_cells,
	const int&               num_wet_cells,
	const bool&              first_t_step
);