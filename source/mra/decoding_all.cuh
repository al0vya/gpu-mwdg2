#pragma once

#include "../utilities/get_num_blocks.h"

#include "../utilities/cuda_utils.cuh"
#include "decoding_kernel.cuh"
#include "decoding_kernel_single_block.cuh"
#include "extra_significance.cuh"

void decoding_all
(
	bool*              d_sig_details,
	real*              d_norm_details,
	Details&           d_details,
	ScaleCoefficients& d_scale_coeffs,
	SolverParams&      solver_params
);