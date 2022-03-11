#pragma once

#include "compact.cuh"

__device__ __forceinline__
Coordinate get_i_index(MortonCode code)
{
	return compact(code);
}