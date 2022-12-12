#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"

__device__ __forceinline__
real h_init_three_blocks
(
	const real& z_int
)
{
	return C(1.78) - z_int;
}