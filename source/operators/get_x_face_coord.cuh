#pragma once

#include "cuda_runtime.h"

#include "../types/real.h"
#include "../types/Directions.h"

__device__ __forceinline__
real get_x_face_coord
(
	const real& x,
	const real& dx_loc,
	const int&  direction
)
{
	real x_face = C(0.0);
	
	switch (direction)
	{
		case NORTH:
		case SOUTH:
			x_face = x - dx_loc / C(2.0) + C(0.5) * dx_loc;
			break;
		case EAST:
			x_face = x - dx_loc / C(2.0) + C(1.0) * dx_loc;
			break;
		case WEST:
			x_face = x - dx_loc / C(2.0) + C(0.0) * dx_loc;
			break;
		default:
			break;
	}
	
	return x_face;
}