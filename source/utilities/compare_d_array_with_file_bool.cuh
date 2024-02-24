#pragma once

#include <cstdio>

#include "cuda_utils.cuh"
#include "compare_array_with_file_bool.h"

int compare_d_array_with_file_bool
(
	const char* dirroot,
	const char* filename,
	bool*       d_array,
	const int&  array_length
);