#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "HierarchyIndex.h"
#include "ScaleCoefficients.h"
#include "SolverParameters.h"
#include "Details.h"
#include "Detail.h"
#include "Maxes.h"
#include "ChildScaleCoeffs.h"
#include "ParentScaleCoeffs.h"

#include "store_details.cuh"
#include "store_scale_coeffs.cuh"
#include "get_lvl_idx.cuh"
#include "encode_scale_coeffs.cuh"
#include "encode_details.cuh"

__global__ void encode_and_thresh_topo
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParameters  solver_params,
	int               level,
	bool              first_time_step
);