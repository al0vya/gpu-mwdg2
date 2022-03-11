#pragma once

#include "cuda_runtime.h"

#include "real.h"
#include "HierarchyIndex.h"

#include "get_y_coord.cuh"

__device__ __forceinline__
real get_y_face_unit
(
	const HierarchyIndex& h_idx,
	const HierarchyIndex& h_idx_nghbr,
	const int&            level_nghbr,
	const int&            max_ref_lvl,
	const real&           y_face, 
	const real&           dy_finest, 
	const int&            direction
)
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
	
	real dy_loc_nghbr = dy_finest * ( 1 << (max_ref_lvl - level_nghbr) );

	real y_nghbr = get_y_coord(h_idx_nghbr, level_nghbr, max_ref_lvl, dy_finest);

	return ( y_face - ( y_nghbr - dy_loc_nghbr / C(2.0) ) ) / dy_loc_nghbr;
}