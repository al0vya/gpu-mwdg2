#pragma once

#include <cstdio>

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

real* read_d_array_real
(
	const int&  num_items,
	const char* dirroot,
	const char* filename
);