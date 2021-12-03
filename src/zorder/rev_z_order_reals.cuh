#pragma once

#include "cuda_utils.cuh"

#include "cub/device/device_radix_sort.cuh"

#include "real.h"
#include "MortonCode.h"

__host__
void rev_z_order_reals
(
	MortonCode* d_rev_z_order,
	MortonCode* d_indices,
	real*       d_array,
	real*       d_array_sorted,
	int         array_length
);