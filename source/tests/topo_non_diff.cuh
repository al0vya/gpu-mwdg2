#pragma once

#include "cuda_runtime.h"

#include "real.h"

// from https://www.sciencedirect.com/science/article/pii/S004578251830389X
__device__ __forceinline__
real topo_non_diff
(
	const real& x_int,
	const real& y_int
)
{
	real z_int = 0;

	bool y_bound = ( C(11.0) <= y_int && y_int <= C(19.0) );

	if (C(16.0) <= x_int && x_int <= C(24.0) && y_bound)
	{
		z_int = C(0.86);
	}
	else if (C(36.0) <= x_int && x_int <= C(44.0) && y_bound)
	{
		z_int = C(1.78);
	}
	else if (C(56.0) <= x_int && x_int <= C(64.0) && y_bound)
	{
		z_int = C(2.30);
	}

	return z_int;
}