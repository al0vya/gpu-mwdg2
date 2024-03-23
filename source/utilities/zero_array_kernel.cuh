#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../types/real.h"

__global__
void zero_array_kernel
(
	real* d_array,
	int   num_threads
);