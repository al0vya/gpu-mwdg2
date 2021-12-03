#pragma once

#include "cuda_utils.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "HierarchyIndex.h"
#include "SolverParameters.h"

#include "get_num_blocks.h"

#include "regularisation.cuh"

__host__
void get_reg_tree
(
	bool*            d_sig_details,
	SolverParameters solver_params
);