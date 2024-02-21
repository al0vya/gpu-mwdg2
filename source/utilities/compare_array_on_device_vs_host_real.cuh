#pragma once

#include "are_reals_equal.h"
#include "cuda_utils.cuh"

bool compare_array_on_device_vs_host_real
(
	real*      h_array,
	real*      d_array,
	const int& array_length
);