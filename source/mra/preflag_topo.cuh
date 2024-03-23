#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/SolverParams.h"
#include "../utilities/get_num_blocks.h"
#include "../types/HierarchyIndex.h"

#include "encode_topography_kernel_hw.cuh"
#include "encode_topography_kernel_mw.cuh"

__host__
void preflag_topo
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&      solver_params
);