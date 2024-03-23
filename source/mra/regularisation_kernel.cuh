#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../types/real.h"
#include "../classes/SigChildren.h"
#include "../types/HierarchyIndex.h"

#include "../utilities/get_lvl_idx.cuh"

__global__
void regularisation_kernel
(
	bool* d_sig_details,
	int   level,
	int   num_threads
);