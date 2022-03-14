#pragma once

#include "cub/device/device_reduce.cuh"

#include <cmath>

#include "cuda_utils.cuh"

#include "real.h"

struct AbsMax
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T& a, const T& b) const { return ( abs(a) > abs(b) ) ? abs(a) : abs(b); }
};

__host__
real get_max_from_array
(
	real*      d_array,
	const int& array_length
);