#pragma once

#include <cstdio>

#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

real* read_hierarchy_array_real
(
	const int&  levels,
	const char* dirroot,
	const char* filename
);