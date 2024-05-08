#pragma once

#include "cub/device/device_reduce.cuh"

#include <cmath>

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

__host__
int get_sum_from_array
(
	bool*      d_array,
	const int& array_length
);