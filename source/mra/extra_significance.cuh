#pragma once

#include "../utilities/get_num_blocks.h"

#include "../utilities/cuda_utils.cuh"
#include "extra_significance_kernel.cuh"
#include "extra_significance_kernel_single_block.cuh"

void extra_significance
(
	bool*         d_sig_details,
	real*         d_norm_details,
	SolverParams& solver_params
);