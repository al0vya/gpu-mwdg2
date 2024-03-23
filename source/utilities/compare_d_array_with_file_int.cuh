#pragma once

#include <cstdio>

#include "cuda_utils.cuh"
#include "compare_array_with_file_int.h"

int compare_d_array_with_file_int
(
	const char* dirroot,
	const char* filename,
	int*        d_array,
	const int&  array_length,
	const int&  offset
);