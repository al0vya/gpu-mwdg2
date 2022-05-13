#include "get_y_coord.cuh"

__device__
real get_y_coord
(
	const HierarchyIndex& h_idx, 
	const int&            level, 
	const int&            max_ref_lvl, 
	const real&           dy_finest
)
{
	real dy_loc = dy_finest * ( 1 << (max_ref_lvl - level) );
	
	MortonCode code = h_idx - get_lvl_idx(level);

	Coordinate j = get_j_index(code);

	return get_spatial_coord(j, dy_loc);
}