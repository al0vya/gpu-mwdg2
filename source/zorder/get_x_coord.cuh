#pragma once

#include "cuda_runtime.h"

#include "get_lvl_idx.cuh"
#include "get_i_index.cuh"
#include "get_spatial_coord.cuh"

__device__ __forceinline__
real get_x_coord
(
	const HierarchyIndex& h_idx, 
	const int&            level, 
	const real&           dx_loc
)
{
	MortonCode code = h_idx - get_lvl_idx(level);

	Coordinate i = get_i_index(code);

	return get_spatial_coord(i, dx_loc);
}