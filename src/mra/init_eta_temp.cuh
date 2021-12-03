#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "AssembledSolution.h"

__global__
void init_eta_temp
(
	AssembledSolution d_assem_sol,
	real*             d_eta_temp
);