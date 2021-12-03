#pragma once

#include "cuda_runtime.h"

#include "real.h"

__device__ __forceinline__
real h_init_radial
(
	const real& x_int,
	const real& y_int
)
{
	const real r = sqrt(x_int * x_int + y_int * y_int);

	return ( r < C(2.5) ) ? C(2.5) : C(0.5);
}