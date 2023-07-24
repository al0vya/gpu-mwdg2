#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../../cub/block/block_store.cuh"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../types/HierarchyIndex.h"
#include "../classes/AssembledSolution.h"
#include "../types/MortonCode.h"
#include "../classes/SolverParams.h"
#include "../classes/ScaleCoefficients.h"

#include "../utilities/get_lvl_idx.cuh"

__global__
void traverse_tree_of_sig_details
(
	bool*             d_sig_details,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_buf_assem_sol,
	int               num_threads,
	SolverParams      solver_params
);