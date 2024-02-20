#pragma once

#include <cstdio>

#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

int* read_hierarchy_array_int
(
	const int&  levels,
	const char* dirroot,
	const char* filename
);