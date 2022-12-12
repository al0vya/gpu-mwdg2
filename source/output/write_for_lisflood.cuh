#pragma once

#include <cstdio>
#include <cstring>

#include "../utilities/BLOCK_VAR_MACROS.cuh"
#include "../utilities/cuda_utils.cuh"
#include "../classes/AssembledSolution.h"
#include "../classes/SimulationParams.h"

__host__ void write_for_lisflood
(
	const char*                 respath,
	const AssembledSolution&    d_assem_sol,
	const int&                  mesh_dim,
	const real&                 dx_finest,
	const SimulationParams& sim_params
);