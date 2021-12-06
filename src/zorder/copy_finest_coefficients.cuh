#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "AssembledSolution.h"
#include "ScaleCoefficients.h"
#include "SolverParams.h"

__global__
void copy_finest_coefficients
(
	AssembledSolution d_assem_sol,
	ScaleCoefficients d_scale_coeffs,
	SolverParams  solver_params,
	HierarchyIndex    finest_lvl_idx
);