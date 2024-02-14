#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"

__device__ __forceinline__
real topo_parabolic_bowl
(
	const real& x_or_y_int
)
{
	return C(0.01) * x_or_y_int * x_or_y_int;
}