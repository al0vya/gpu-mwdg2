#pragma once

#include "../zorder/compact.cuh"

__host__ __device__ __forceinline__
Coordinate get_i_index(MortonCode code)
{
	return compact(code);
}