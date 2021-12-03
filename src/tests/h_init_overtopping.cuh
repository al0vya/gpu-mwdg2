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
	real eta_west = bcs.hl;
	real eta_east = bcs.hr;

	real h = (x_or_y_int <= 25) ? eta_west - z_int : ( eta_east - z_int < C(0.0) ) ? eta_east - z_int : eta_east - z_int;

	return (x_or_y_int <= 25) ? max(C(0.0), bcs.hl - z_int) : max(C(0.0), bcs.hr - z_int);
}