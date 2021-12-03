#pragma once

#include "SolverParameters.h"
#include "AssembledSolution.h"

#include "write_cumu_sim_time.h"

#include "append_reals_to_file.cuh"

__host__
void write_c_prop_data
(
	const char*              respath,
	const clock_t&           start,
	const SolverParameters&  solver_params,
	const AssembledSolution& d_assem_sol,
	const real&              time_now,
	const bool&              first_t_step
);