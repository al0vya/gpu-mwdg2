#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "SolverParams.h"
#include "HierarchyIndex.h"
#include "Details.h"
#include "ScaleCoefficients.h"
#include "BLOCK_VAR_MACROS.cuh"

#include "get_lvl_idx.cuh"
#include "load_details.cuh"
#include "load_parent_scale_coeffs.cuh"
#include "decode_scale_coeffs.cuh"
#include "store_scale_coeffs.cuh"

template <bool SINGLE_BLOCK>
__global__
void decoding
(
	bool*             d_sig_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams  solver_params,
	int               level,
	int               num_threads
);