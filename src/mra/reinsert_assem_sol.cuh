#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ScaleCoefficients.h"
#include "AssembledSolution.h"
#include "SolverParameters.h"

__global__
void reinsert_assem_sol
(
	AssembledSolution d_assem_sol,
	HierarchyIndex*   act_idcs, 
	ScaleCoefficients d_scale_coeffs,
	SolverParameters  solver_params
);