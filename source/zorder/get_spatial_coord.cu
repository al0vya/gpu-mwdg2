#include "get_spatial_coord.cuh"

__device__
real get_spatial_coord
(
	const int&  idx,
	const real& cellsize
)
{
	return idx * cellsize + cellsize / C(2.0);
}