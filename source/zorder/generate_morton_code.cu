#include "generate_morton_code.cuh"

__host__ __device__
MortonCode generate_morton_code
(
	Coordinate x,
	Coordinate y
)
{
	// please grep 'BITWISE OPERATORS AND HEXADECIMAL' for explanation
	return dilate(x) | (dilate(y) << 1);
}