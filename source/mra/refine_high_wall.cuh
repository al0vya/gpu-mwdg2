#pragma once

#include "cuda_utils.cuh"

#include "SimulationParams.h"
#include "SolverParams.h"

#include "generate_morton_code.cuh"
#include "get_lvl_idx.cuh"

__host__
void refine_high_wall
(
	const SimulationParams& sim_params,
	const SolverParams&     solver_params,
	const int               max_ref_lvl,
	      bool*             h_preflagged_details
);