#include "dilate.cuh"

__host__ __device__
MortonCode dilate(Coordinate coord)
{

	// first, truncate to 16 bits so that later,
	// when generating Morton codes from 2 coordinates,
	// this ensures each code is 16 + 16 = 32 bits long

	coord &= 0x0000ffff;                         // in binary: ---- ---- ---- ---- fedc ba98 7654 3210

	coord = (coord ^ (coord << 8)) & 0x00ff00ff; // in binary: ---- ---- fedc ba98 ---- ---- 7654 3210
	coord = (coord ^ (coord << 4)) & 0x0f0f0f0f; // in binary: ---- fedc ---- ba98 ---- 7654 ---- 3210
	coord = (coord ^ (coord << 2)) & 0x33333333; // in binary: --fe --dc --ba --98 --76 --54 --32 --10
	coord = (coord ^ (coord << 1)) & 0x55555555; // in binary: -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0

	return coord;
}