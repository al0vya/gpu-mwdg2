#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../types/real.h"
#include "../classes/DetailChildren.h"
#include "../types/HierarchyIndex.h"

#include "get_child_details.cuh"
#include "../utilities/get_lvl_idx.cuh"

template <bool SINGLE_BLOCK>
__global__ void regularisation
(
	bool*    d_sig_details,
	int      level,
	int      num_threads
);