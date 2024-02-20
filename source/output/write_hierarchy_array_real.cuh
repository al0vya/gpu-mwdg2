#pragma once

#include <cstdio>
#include <cstring>

#include "../types/real.h"

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

__host__
void write_hierarchy_array_real
(
	const char* dirroot,
	const char* filename,
	real*       d_hierarchy,
	const int&  levels
);