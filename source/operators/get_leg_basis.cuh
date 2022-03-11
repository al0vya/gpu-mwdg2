#include "cuda_runtime.h"

#include "LegendreBasis.h"

#include "get_x_coord.cuh"
#include "get_y_coord.cuh"
#include "get_x_face_coord.cuh"
#include "get_y_face_coord.cuh"
#include "get_x_face_unit.cuh"
#include "get_y_face_unit.cuh"

__device__ __forceinline__
LegendreBasis get_leg_basis
(
	const HierarchyIndex& h_idx,
	const real&           x,
	const real&           y,
	const real&           dx_loc,
	const real&           dy_loc,
	const int&            direction
)
{
	real x_face = get_x_face_coord(x, dx_loc, direction);
	real y_face = get_y_face_coord(y, dy_loc, direction);

	real x_face_unit = get_x_face_unit(h_idx, x, x_face, dx_loc, direction);
	real y_face_unit = get_y_face_unit(h_idx, y, y_face, dy_loc, direction);
	
	return
	{
		C(1.0),
		sqrt( C(3.0) ) * ( C(2.0) * x_face_unit - C(1.0) ),
		sqrt( C(3.0) ) * ( C(2.0) * y_face_unit - C(1.0) )
	};
}