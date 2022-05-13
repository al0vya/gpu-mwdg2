#include "get_i_index.cuh"

__host__ __device__
Coordinate get_i_index(MortonCode code)
{
	return compact(code);
}