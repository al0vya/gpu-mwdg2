#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_store.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "HierarchyIndex.h"
#include "AssembledSolution.h"
#include "MortonCode.h"
#include "SolverParams.h"
#include "ScaleCoefficients.h"

#include "get_lvl_idx.cuh"

__global__
void traverse_tree_of_sig_details
(
	bool*             d_sig_details,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_buf_assem_sol,
	int               num_threads,
	SolverParams  solver_params
);