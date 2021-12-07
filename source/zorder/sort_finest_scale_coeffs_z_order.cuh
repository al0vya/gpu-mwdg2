#pragma once

#include "cuda_utils.cuh"

#include "cub/cub.cuh"

#include "CHECK_CUDA_ERROR.cuh"

#include "AssembledSolution.h"
#include "SolverParams.h"
#include "MortonCode.h"

__host__
void sort_finest_scale_coeffs_z_order
(
	MortonCode*        d_morton_codes,
	MortonCode*        d_sorted_morton_codes,
	AssembledSolution& d_assem_sol,
	AssembledSolution& d_buf_assem_sol,
	MortonCode*        d_indices,
	MortonCode*        d_rev_z_order,
	SolverParams&  solver_params
);