#pragma once

#include "cuda_utils.cuh"

#include "cub/cub.cuh"

#include "AssembledSolution.h"
#include "MortonCode.h"

__host__
void rev_z_order_act_idcs
(
	MortonCode*       d_rev_z_order,
	MortonCode*       d_indices,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	int               array_length
);