#pragma once

#include <stdio.h>
#include <string.h>

#include "BLOCK_VAR_MACROS.cuh"
#include "cuda_utils.cuh"
#include "AssembledSolution.h"
#include "SimulationParameters.h"

__host__ void write_for_lisflood
(
	const char*                 respath,
	const AssembledSolution&    d_assem_sol,
	const int&                  mesh_dim,
	const real&                 dx_finest,
	const SimulationParameters& sim_params
);