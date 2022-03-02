#pragma once

#include "cuda_utils.cuh"

#include "SolverParams.h"
#include "get_num_blocks.h"
#include "HierarchyIndex.h"

#include "encode_and_thresh_topo.cuh"

__host__
void preflag_topo
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&  solver_params,
	SimulationParams& sim_params,
	int                first_time_step
);