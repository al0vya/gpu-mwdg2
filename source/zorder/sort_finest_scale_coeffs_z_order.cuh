#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../../cub/cub.cuh"

#include "../utilities/CHECK_CUDA_ERROR.cuh"

#include "../classes/AssembledSolution.h"
#include "../classes/SolverParams.h"
#include "../types/MortonCode.h"

__global__
void sort_finest_scale_coeffs_z_order
(
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	MortonCode*       d_rev_z_order,
	SolverParams      solver_params
);