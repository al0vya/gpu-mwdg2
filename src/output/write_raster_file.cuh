#pragma once

#include <stdio.h>
#include <string.h>

#include "BLOCK_VAR_MACROS.cuh"
#include "cuda_utils.cuh"

#include "SimulationParameters.h"
#include "SaveInterval.h"

__host__
void write_raster_file
(
	const char*                 respath,
	const char*                 file_extension,
	real*                       d_raster_array,
	const SimulationParameters& sim_params,
	const SaveInterval          massint,
	const real&                 dx_finest,
	const int&                  mesh_dim
);