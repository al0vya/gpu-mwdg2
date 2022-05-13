#pragma once

#include "cuda_runtime.h"

#include "get_lvl_idx.cuh"
#include "get_i_index.cuh"
#include "get_spatial_coord.cuh"

__device__
real get_x_coord
(
	const HierarchyIndex& h_idx, 
	const int&            level, 
	const int&            max_ref_lvl, 
	const real&           dx_finest
);