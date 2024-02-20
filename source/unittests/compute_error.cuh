#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_num_blocks.h"
#include "../unittests/compute_error_kernel.cuh"
#include "../mra/get_mean_from_array.cuh"

real compute_error
(
	const char* dirroot,
	const char* filename,
	real*       d_computed,
	real*       d_verified,
	const int&  array_length
);