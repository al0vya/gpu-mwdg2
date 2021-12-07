#pragma once 

#include "cub/device/device_reduce.cuh"

#include "cuda_utils.cuh"

#include "real.h"

__host__
real get_dt_CFL
(
	real*&     d_dt_CFL, 
	const int& sol_len
);