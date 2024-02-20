#pragma once

#include <cstdio>
#include <cstring>

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

__host__
void write_hierarchy_array_bool
(
	const char* dirroot,
	const char* filename,
	bool*       d_hierarchy,
	const int&  levels
);