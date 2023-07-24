#pragma once 

#include "cub/device/device_reduce.cuh"

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

__host__
real get_dt_CFL
(
	real*&     d_dt_CFL, 
	const int& sol_len
);