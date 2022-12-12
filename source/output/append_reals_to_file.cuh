#pragma once

#include "../utilities/cuda_utils.cuh"

#include <cstdio>
#include <cstring>

#include "../types/real.h"


__host__
void append_reals_to_file
(
	const char* filename,
	const char* respath,
	real*       d_results,
	const int&  array_length,
	const bool  first_t_step
);