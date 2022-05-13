#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "Directions.h"

__device__ __forceinline__
real get_y_face_coord
(
	const real& y,
	const real& dy_loc,
	const int&  direction
)
{
	real y_face = C(0.0);
	
	switch (direction)
	{
		case NORTH:
			y_face = y - dy_loc / C(2.0) + C(1.0) * dy_loc;
			break;
		case EAST:
		case WEST:
			y_face = y - dy_loc / C(2.0) + C(0.5) * dy_loc;
			break;
		case SOUTH:
			y_face = y - dy_loc / C(2.0) + C(0.0) * dy_loc;
			break;
		default:
			break;
	}
	
	return y_face;
}