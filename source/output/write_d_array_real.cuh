#pragma once

#include <cstdio>
#include <cstring>

#include "../types/real.h"

#include "../utilities/cuda_utils.cuh"

__host__
void write_d_array_real
(
	const char* dirroot,
	const char* filename,
	real*       d_array,
	const int&  array_length
);