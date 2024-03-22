#pragma once

#include <algorithm>

#include "cuda_utils.cuh"

int compare_array_on_device_vs_host_int
(
	int*       h_array,
	int*       d_array,
	const int& array_length
);