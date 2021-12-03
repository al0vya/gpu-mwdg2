#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BLOCK_VAR_MACROS.cuh"

#include "real.h"
#include "DetailChildren.h"
#include "HierarchyIndex.h"

#include "get_child_details.cuh"
#include "get_lvl_idx.cuh"

template <bool SINGLE_BLOCK>
__global__ void regularisation
(
	bool*    d_sig_details,
	int      level,
	int      num_threads
);