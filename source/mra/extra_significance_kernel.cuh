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

__global__
void extra_significance_kernel
(
	bool*          d_sig_details,
	real*          d_norm_details,
	real           eps_local,
	real           eps_extra_sig,
	HierarchyIndex curr_lvl_idx,
	HierarchyIndex next_lvl_idx,
	int            level,
	int            num_threads
);