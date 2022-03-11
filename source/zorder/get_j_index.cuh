#pragma once

#include "compact.cuh"

__device__ __forceinline__
Coordinate get_j_index(MortonCode code)
{
	return compact(code >> 1);
}