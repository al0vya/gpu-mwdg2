#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../../cub/cub.cuh"

#include "../utilities/CHECK_CUDA_ERROR.cuh"

#include "../classes/AssembledSolution.h"
#include "../classes/SolverParams.h"
#include "../types/MortonCode.h"

__host__
void get_sorting_indices
(
	MortonCode*        d_morton_codes,
	MortonCode*        d_sorted_morton_codes,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	MortonCode*        d_indices,
	MortonCode*        d_rev_z_order,
	MortonCode*        d_rev_row_major,
	SolverParams&      solver_params
);