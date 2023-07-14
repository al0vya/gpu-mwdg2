#pragma once

#include "../utilities/cuda_utils.cuh"

#include <cstdio>
#include <cstring>


__host__
void write_int_to_file
(
	const char* filename,
	const char* dirroot,
	int*        d_results,
	const int   array_length
);