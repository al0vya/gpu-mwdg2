#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "HierarchyIndex.h"

__device__ __forceinline__
real get_x_face_unit(const HierarchyIndex h_idx, const real& x, const real& x_face, const real& dx_loc, const int& direction)
{
	if (h_idx == -1)
	{
		real x_face_unit = C(0.0);

		switch (direction)
		{
			case NORTH:
			case SOUTH:
				x_face_unit = C(0.5);
				break;
			case EAST:
				x_face_unit = C(0.0);
				break;
			case WEST:
				x_face_unit = C(1.0);
				break;
			default:
				break;
		}

		return x_face_unit;
	}

	return ( x_face - ( x - dx_loc / C(2.0) ) ) / dx_loc;
}