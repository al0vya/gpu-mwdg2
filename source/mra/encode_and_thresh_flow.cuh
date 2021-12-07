#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"
#include "HierarchyIndex.h"
#include "ScaleCoefficients.h"
#include "SolverParams.h"
#include "Details.h"
#include "Detail.h"
#include "Maxes.h"
#include "ChildScaleCoeffs.h"
#include "ParentScaleCoeffs.h"

#include "get_lvl_idx.cuh"
#include "store_details.cuh"
#include "store_scale_coeffs.cuh"
#include "encode_scale_coeffs.cuh"
#include "encode_details.cuh"

template<bool SINGLE_BLOCK>
__global__
void encode_and_thresh_flow
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams  solver_params,
	int               level,
	int               num_threads,
	bool              for_nghbrs
);