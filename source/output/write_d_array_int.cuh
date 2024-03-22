#pragma once

#include <cstdio>
#include <cstring>

#include "../utilities/cuda_utils.cuh"

__host__
void write_d_array_int
(
	const char* dirroot,
	const char* filename,
	int*        d_array,
	const int&  array_length
);