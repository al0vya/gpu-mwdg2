#pragma once

#include "BLOCK_VAR_MACROS.cuh"

#include "AssembledSolution.h"
#include "Maxes.h"

#include "get_num_blocks.h"

#include "init_eta_temp.cuh"
#include "get_max_from_array.cuh"

__host__
Maxes get_max_scale_coeffs
(
	AssembledSolution& d_assem_sol,
	real*&             d_eta_temp
);