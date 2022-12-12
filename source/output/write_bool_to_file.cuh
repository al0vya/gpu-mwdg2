#pragma once

#include "../utilities/cuda_utils.cuh"

#include <cstdio>
#include <cstring>

__host__
void write_bool_to_file
(
	const char* filename,
	const char* respath,
	bool*       d_results,
	const int   array_length
);