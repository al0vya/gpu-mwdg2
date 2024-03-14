#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../types/HierarchyIndex.h"
#include "../classes/SolverParams.h"

#include "../utilities/get_num_blocks.h"

#include "regularisation_kernel.cuh"
#include "regularisation_kernel_single_block.cuh"

__host__
void get_reg_tree
(
	bool*        d_sig_details,
	SolverParams solver_params
);