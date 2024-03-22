#pragma once

#include <cstdio>

#include "../utilities/cuda_utils.cuh"

int* read_d_array_int
(
	const int&  num_items,
	const char* dirroot,
	const char* filename
);