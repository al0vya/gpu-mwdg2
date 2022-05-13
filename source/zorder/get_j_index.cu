#include "get_j_index.cuh"

__host__ __device__
Coordinate get_j_index(MortonCode code)
{
	return compact(code >> 1);
}