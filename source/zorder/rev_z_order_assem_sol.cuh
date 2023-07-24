#pragma once

#include "cuda_runtime.h"
#include "cub/cub.cuh"

#include "../utilities/cuda_utils.cuh"

#include "../classes/AssembledSolution.h"
#include "../types/MortonCode.h"

void rev_z_order_assem_sol
(
	MortonCode*       d_rev_z_order,
	MortonCode*       d_indices,
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	int               array_length
);