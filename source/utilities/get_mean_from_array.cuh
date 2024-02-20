#pragma once

#include "cub/device/device_reduce.cuh"

#include <cmath>

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

__host__
real get_mean_from_array
(
	real*      d_array,
	const int& array_length
);