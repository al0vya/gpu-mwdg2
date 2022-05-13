#pragma once

#include "cuda_runtime.h"

#include "MortonCode.h"
#include "Coordinate.h"

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
__host__ __device__
MortonCode dilate(Coordinate coord);