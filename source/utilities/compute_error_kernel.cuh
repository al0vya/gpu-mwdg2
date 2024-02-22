#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../types/real.h"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

__global__
void compute_error_kernel
(
	real* d_computed,
	real* d_verified,
	real* d_errors,
	int   array_length
);