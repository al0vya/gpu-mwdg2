#pragma once

#include "cuda_utils.cuh"

#include "cub/cub.cuh"

#include "AssembledSolution.h"
#include "MortonCode.h"

__global__
void rev_z_order_act_idcs
(
	MortonCode*       d_rev_row_major,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	const int         num_finest_elems
);