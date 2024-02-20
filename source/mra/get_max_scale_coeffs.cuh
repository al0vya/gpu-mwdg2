#pragma once

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../classes/AssembledSolution.h"
#include "../classes/Maxes.h"

#include "../utilities/get_num_blocks.h"
#include "../utilities/get_max_from_array.cuh"

#include "../mra/init_eta_temp.cuh"

__host__
Maxes get_max_scale_coeffs
(
	AssembledSolution& d_assem_sol,
	real*&             d_eta_temp
);