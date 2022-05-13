#include "get_leg_basis.cuh"

__device__
LegendreBasis get_leg_basis
(
	const HierarchyIndex& h_idx,
	const HierarchyIndex& h_idx_nghbr,
	const int&            level_nghbr,
	const int&            max_ref_lvl,
	const real&           x,
	const real&           y,
	const real&           dx_loc,
	const real&           dy_loc,
	const real&           dx_finest,
	const real&           dy_finest,
	const int&            direction
)
{
	real x_face = get_x_face_coord(x, dx_loc, direction);
	real y_face = get_y_face_coord(y, dy_loc, direction);

	real x_face_unit = get_x_face_unit(h_idx, h_idx_nghbr, level_nghbr, max_ref_lvl, x_face, dx_finest, direction);
	real y_face_unit = get_y_face_unit(h_idx, h_idx_nghbr, level_nghbr, max_ref_lvl, y_face, dy_finest, direction);

	return
	{
		C(1.0),
		sqrt( C(3.0) ) * ( C(2.0) * x_face_unit - C(1.0) ),
		sqrt( C(3.0) ) * ( C(2.0) * y_face_unit - C(1.0) )
	};
}