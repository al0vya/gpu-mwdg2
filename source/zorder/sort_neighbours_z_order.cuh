#pragma once

#include "cuda_runtime.h"

#include "cub/device/device_radix_sort.cuh"

#include "CHECK_CUDA_ERROR.cuh"
#include "cuda_utils.cuh"

#include "Neighbours.h"
#include "MortonCode.h"
#include "SolverParams.h"

__global__
void sort_neighbours_z_order
(
	const Neighbours   d_neighbours,
	const Neighbours   d_buf_neighbours,
	MortonCode*        d_rev_z_order,
	const int          num_finest_elems,
	const SolverParams solver_params
);