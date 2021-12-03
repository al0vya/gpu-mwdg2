#pragma once

#include "cuda_utils.cuh"

#include "SolverParameters.h"
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
	SolverParameters&  solver_params,
	int                first_time_step
);