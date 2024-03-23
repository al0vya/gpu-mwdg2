#pragma once

#include "../classes/Details.h"
#include "../classes/SolverParams.h"
#include "../types/HierarchyIndex.h"
#include "../utilities/zero_array_kernel_real.cuh"
#include "../utilities/get_num_blocks.h"

void zero_details
(
	Details      d_details,
	real*        d_norm_details,
	int          num_details,
	SolverParams solver_params
);