#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"

// from https://www.sciencedirect.com/science/article/pii/S0309170815002237
__device__ __forceinline__
real topo_triangle
(
	const real& x_or_y_int
)
{
	real slope = C(0.4) / C(3.0);
	
	real z_int = C(0.0);

	if ( x_or_y_int > C(25.5) && x_or_y_int <= C(28.5) )
	{
		z_int = ( x_or_y_int - C(25.5) ) * slope;
	}
	else if ( x_or_y_int > C(28.5) && x_or_y_int <= C(31.5) )
	{
		z_int = C(0.4) - ( x_or_y_int - C(28.5) ) * slope;
	}

	return z_int;
}