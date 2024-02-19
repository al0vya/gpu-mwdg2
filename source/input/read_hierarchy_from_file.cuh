#pragma once

#include <cstdio>

#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

void read_hierarchy_from_file
(
	      real* d_hierarchy,
	const int&  levels,
	const char* dirroot,
	const char* filename
);