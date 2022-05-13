#include "get_x_coord.cuh"

__device__
real get_x_coord
(
	const HierarchyIndex& h_idx, 
	const int&            level, 
	const int&            max_ref_lvl, 
	const real&           dx_finest
)
{
	real dx_loc = dx_finest * ( 1 << (max_ref_lvl - level) );
	
	MortonCode code = h_idx - get_lvl_idx(level);

	Coordinate i = get_i_index(code);

	return get_spatial_coord(i, dx_loc);
}