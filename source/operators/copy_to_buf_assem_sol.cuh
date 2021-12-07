#pragma once

#include "cuda_utils.cuh"

#include "AssembledSolution.h"

__host__
void copy_to_buf_assem_sol
(
	const AssembledSolution& d_assem_sol,
	AssembledSolution&       d_buf_assem_sol
);