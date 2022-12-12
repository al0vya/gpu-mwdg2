#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../classes/ScaleCoefficients.h"
#include "../classes/PointSources.h"

#include "../utilities/get_lvl_idx.cuh"

__global__
void reinsert_point_srcs
(
	ScaleCoefficients d_scale_coeffs,
	PointSources      point_sources,
	real              dt,
	real              dx_finest,
	int               max_ref_lvl
);