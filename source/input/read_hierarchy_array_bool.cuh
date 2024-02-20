#pragma once

#include <cstdio>

#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/cuda_utils.cuh"

bool* read_hierarchy_array_bool
(
	const int&  levels,
	const char* dirroot,
	const char* filename
);