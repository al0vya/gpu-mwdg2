#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Details.h"
#include "HierarchyIndex.h"
#include "SolverParams.h"

__global__
void zero_details
(
	Details          d_details,
	real*            d_norm_details,
	int              num_details,
	SolverParams solver_params
);