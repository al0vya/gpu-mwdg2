#pragma once

#include <algorithm>

#include "are_reals_equal.h"
#include "cuda_utils.cuh"

real compare_array_on_device_vs_host_real
(
	real*      h_array,
	real*      d_array,
	const int& array_length
);