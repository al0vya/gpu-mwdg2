#pragma once

#include "cuda_runtime.h"

#include "real.h"

// from https://www.sciencedirect.com/science/article/pii/S004578251830389X
__device__ __forceinline__
real diff_topo
(
	const real& x_int,
	const real& y_int
)
{
	real z_int = 0;

	real cone1 = C(1.0) - C(0.2) * sqrt( ( x_int - C(20.0) ) * ( x_int - C(20.0) ) + ( y_int - C(15.0) ) * ( y_int - C(15.0) ) );
	real cone2 = C(2.0) - C(0.5) * sqrt( ( x_int - C(40.0) ) * ( x_int - C(40.0) ) + ( y_int - C(15.0) ) * ( y_int - C(15.0) ) );
	real cone3 = C(3.0) - C(0.3) * sqrt( ( x_int - C(60.0) ) * ( x_int - C(60.0) ) + ( y_int - C(15.0) ) * ( y_int - C(15.0) ) );

	z_int = max(z_int, cone1);
	z_int = max(z_int, cone2);
	z_int = max(z_int, cone3);

	return z_int;
}