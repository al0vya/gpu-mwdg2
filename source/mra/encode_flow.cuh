#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/Tracer.h"
#include "../classes/SolverParams.h"
#include "../types/HierarchyIndex.h"

#include "../utilities/get_num_blocks.h"

#include "encode_flow_kernel_hw.cuh"
#include "encode_flow_kernel_single_block_hw.cuh"
#include "encode_flow_kernel_mw.cuh"
#include "encode_flow_kernel_single_block_mw.cuh"

__host__
void encode_flow
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	real*              d_norm_details,
	bool*              d_sig_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&      solver_params,
	bool               for_nghbrs
);