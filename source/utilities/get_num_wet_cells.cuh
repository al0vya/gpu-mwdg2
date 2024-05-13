#pragma once

#include "cub/device/device_reduce.cuh"

#include <cmath>

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

struct CountWet
{
	const real tol_h;
    
	CountWet(const real& tol_h) : tol_h(tol_h) {};
    
    template <typename T>
	__device__ __forceinline__
	T operator()(const T& a, const T& b) const { return (a > tol_h) + (b > tol_h); }
};

__host__
real get_num_wet_cells
(
	real*       d_h0,
	const int&  array_length,
    const real& tol_h
);