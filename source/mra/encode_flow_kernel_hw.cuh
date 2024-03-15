#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "../utilities/BLOCK_VAR_MACROS.cuh"
#include "../types/HierarchyIndex.h"
#include "../classes/ScaleCoefficients.h"
#include "../classes/SolverParams.h"
#include "../classes/Details.h"
#include "../classes/Detail.h"
#include "../classes/Maxes.h"
#include "../classes/ChildScaleCoeffs.h"
#include "../classes/ParentScaleCoeffs.h"

#include "../utilities/get_lvl_idx.cuh"
#include "load_children_vector.cuh"
#include "store_details.cuh"
#include "store_scale_coeffs.cuh"
#include "encode_scale_coeffs.cuh"
#include "encode_details.cuh"

__global__
void encode_flow_kernel_hw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	real*             d_norm_details,
	bool*             d_sig_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams      solver_params,
	int               level,
	int               num_threads,
	bool              for_nghbrs
);