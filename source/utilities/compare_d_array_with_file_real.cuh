#pragma once

#include <cstdio>

#include "cuda_utils.cuh"
#include "compare_array_with_file_real.h"

real compare_d_array_with_file_real
(
	const char* dirroot,
	const char* filename,
	real*       d_array,
	const int&  array_length,
	const int&  offset
);