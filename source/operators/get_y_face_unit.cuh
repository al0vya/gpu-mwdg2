#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "HierarchyIndex.h"

__device__ __forceinline__
real get_y_face_unit(const HierarchyIndex h_idx, const real& y, const real& y_face, const real& dy_loc, const int& direction)
{
	if (h_idx == -1)
	{
		real y_face_unit = C(0.0);

		switch (direction)
		{
			case NORTH:
				y_face_unit = C(0.0);
				break;
			case EAST:
			case WEST:
				y_face_unit = C(0.5);
				break;
			case SOUTH:
				y_face_unit = C(1.0);
				break;
			default:
				break;
		}

		return y_face_unit;
	}

	return ( y_face - ( y - dy_loc / C(2.0) ) ) / dy_loc;
}