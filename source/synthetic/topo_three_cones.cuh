#pragma once

#include "cuda_runtime.h"

#include <math.h>

#include "../types/real.h"

__device__ __forceinline__
real topo_three_cones
(
	real x_int,
	real y_int
)
{
	
	real z_int = 0;

	real x_1 = 30;
	real y_1 = 6;
	real x_2 = 30;
	real y_2 = 24;
	real x_3 = C(47.5);
	real y_3 = 15;

	int rm_1 = 8;
	int rm_2 = 8;
	int rm_3 = 10;

	real r_1 = sqrt( (x_int - x_1) * (x_int - x_1) + (y_int - y_1) * (y_int - y_1) );
	real r_2 = sqrt( (x_int - x_2) * (x_int - x_2) + (y_int - y_2) * (y_int - y_2) );
	real r_3 = sqrt( (x_int - x_3) * (x_int - x_3) + (y_int - y_3) * (y_int - y_3) );

	real zb_1 = (rm_1 - r_1) / 8;
	real zb_2 = (rm_2 - r_2) / 8;
	real zb_3 = C(0.3) * (rm_3 - r_3);

	z_int = max(zb_1 , zb_2);
	z_int = max(z_int, zb_3);
	z_int = max(z_int, C(0.0) );

	return z_int;
}