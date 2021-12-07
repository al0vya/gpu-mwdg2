#pragma once

#include "cub/device/device_reduce.cuh"

#include <cmath>

#include "cuda_utils.cuh"
#include "BLOCK_VAR_MACROS.cuh"

#include "AssembledSolution.h"
#include "Maxes.h"

#include "get_num_blocks.h"

#include "init_eta_temp.cuh"

struct AbsMax
{
	template <typename T>
	__device__ __forceinline__
	T operator()(const T& a, const T& b) const { return ( abs(a) > abs(b) ) ? abs(a) : abs(b); }
};

__host__
Maxes get_max_scale_coeffs
(
	AssembledSolution& d_assem_sol,
	real*&             d_eta_temp
);