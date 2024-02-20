#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_num_blocks.h"
#include "../utilities/compute_error_kernel.cuh"
#include "../utilities/get_mean_from_array.cuh"

real compute_error
(
	real*       d_computed,
	real*       d_verified,
	const int&  array_length
);