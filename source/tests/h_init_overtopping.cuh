#pragma once

#include "cuda_runtime.h"

#include "Depths1D.h"

__device__ __forceinline__
real h_init_overtopping
(
	const Depths1D& bcs, 
	const real&               z_int, 
	const real&               x_or_y_int
)
{
	return (x_or_y_int <= 25) ? max(C(0.0), bcs.hl - z_int) : max(C(0.0), bcs.hr - z_int);
}