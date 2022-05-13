#include "get_x_face_unit.cuh"

__device__
real get_x_face_unit
(
	const HierarchyIndex& h_idx,
	const HierarchyIndex& h_idx_nghbr,
	const int&            level_nghbr,
	const int&            max_ref_lvl,
	const real&           x_face, 
	const real&           dx_finest, 
	const int&            direction
)
{
	if (h_idx_nghbr == -1)
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
	
	real dx_loc_nghbr = dx_finest * ( 1 << (max_ref_lvl - level_nghbr) );

	real x_nghbr = get_x_coord(h_idx_nghbr, level_nghbr, max_ref_lvl, dx_finest);

	return ( x_face - ( x_nghbr - dx_loc_nghbr / C(2.0) ) ) / dx_loc_nghbr;
}