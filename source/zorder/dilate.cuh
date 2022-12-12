#pragma once

#include "cuda_runtime.h"

#include "../types/MortonCode.h"
#include "../types/Coordinate.h"

/*
 * BITWISE OPERATORS AND HEXADECIMAL
 *
 * This function uses hexadecimal numbers, which in C++ start with '0x', and bitwise operators.
 * Hexadecimal numbers go from (in binary) 1 = 0000 to f = 1111.
 * For example, 0x0000ffff = 00000000000000001111111111111111 (16 0's, 16 1's).
 * Bitwise operators in C++ are as follows:
 *
 *          OR : |
 *         AND : &
 *         XOR : ^
 *  LEFT SHIFT : <<
 * RIGHT SHIFT : >>
 */

 // inserts a 0 between each bit
 // this blog post is very helpful: http://asgerhoedt.dk/?p=276
__host__ __device__ __forceinline__
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