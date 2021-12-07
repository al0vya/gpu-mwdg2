#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__ __forceinline__
real h_init_three_peaks
(
	const real& z_int
)
{
	return C(1.95) - z_int;
}