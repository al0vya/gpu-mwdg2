#pragma once

#include "cuda_utils.cuh"

#include "SolverParameters.h"
#include "HierarchyIndex.h"

#include "get_num_blocks.h"

#include "encode_and_thresh_flow.cuh"


__host__
void encoding_all
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	real*              d_norm_details,
	bool*              d_sig_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParameters&  solver_params,
	bool               for_nghbrs
);