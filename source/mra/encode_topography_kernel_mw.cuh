#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "../utilities/BLOCK_VAR_MACROS.cuh"
#include "../utilities/get_lvl_idx.cuh"
#include "../types/MortonCode.h"
#include "../types/Coordinate.h"
#include "../classes/SolverParams.h"
#include "../classes/SimulationParams.h"
#include "../classes/Maxes.h"
#include "store_details.cuh"
#include "store_scale_coeffs.cuh"
#include "encode_scale_coeffs.cuh"
#include "encode_details.cuh"
#include "../zorder/compact.cuh"
#include "../zorder/generate_morton_code.cuh"

__global__
void encode_topography_kernel_mw
(
	ScaleCoefficients d_scale_coeffs,
	Details           d_details,
	bool*             d_preflagged_details,
	Maxes             maxes,
	SolverParams      solver_params,
	int               level
);