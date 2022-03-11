#pragma once

#include "cuda_runtime.h"

#include "get_lvl_idx.cuh"
#include "get_j_index.cuh"
#include "get_spatial_coord.cuh"

__device__ __forceinline__
real get_y_coord
(
	const HierarchyIndex& h_idx, 
	const int&            level, 
	const real&           dy_loc
)
{
	MortonCode code = h_idx - get_lvl_idx(level);

	Coordinate j = get_j_index(code);

	return get_spatial_coord(j, dy_loc);
}