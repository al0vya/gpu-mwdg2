#pragma once

#include "cuda_runtime.h"

#include "cuda_utils.cuh"
#include "BLOCK_VAR_MACROS.cuh"

#include "SimulationParams.h"

#include "get_num_blocks.h"
#include "init_nodal_values.cuh"

__host__ void get_nodal_values
(
	NodalValues&                d_nodal_vals,
	const real&                 dx_finest,
	const real&                 dy_finest,
	const Depths1D&             bcs,
	const SimulationParams& sim_params,
	const int&                  interface_dim,
	const int&                  test_case
);