#pragma once

#include "cuda_runtime.h"

#include "../classes/Depths1D.h"

__device__ __forceinline__
real h_init_c_property
(
	const Depths1D& bcs, 
	const real&     z_int,
	const real&     x_or_y_int
)
{
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h = eta_west - z_int;

	return h;

	return (h < 0) ? 0 : h;

	return (x_or_y_int <= 25) ? ( (h < 0) ? bcs.hl : h ) : eta_east - z_int;
}