#pragma once

#ifndef M_BAR
	#define M_BAR C(1.5)
#endif // !M_BAR


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../classes/SolverParams.h"
#include "../types/HierarchyIndex.h"

#include "../utilities/get_lvl_idx.cuh"

template<bool SINGLE_BLOCK>
__global__
void extra_significance
(
	bool*            d_sig_details,
	real*            d_norm_details,
	SolverParams solver_params,
	int              level,
	int              num_threads
);