#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"

#include "../zorder/generate_morton_code.cuh"
#include "../utilities/get_lvl_idx.cuh"

__host__
void refine_high_wall
(
	const SimulationParams& sim_params,
	const SolverParams&     solver_params,
	const int               max_ref_lvl,
	      bool*             h_preflagged_details
);