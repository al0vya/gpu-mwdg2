#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"

__device__ __forceinline__
real get_spatial_coord
(
	const int&  idx,
	const real& cellsize
)
{
	return idx * cellsize + cellsize / C(2.0);
}