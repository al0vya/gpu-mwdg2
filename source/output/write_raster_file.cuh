#pragma once

#include <cstdio>
#include <cstring>

#include "BLOCK_VAR_MACROS.cuh"
#include "cuda_utils.cuh"

#include "SimulationParams.h"
#include "SaveInterval.h"

__host__
void write_raster_file
(
	const char*             respath,
	const char*             file_extension,
	real*                   raster,
	const SimulationParams& sim_params,
	const SaveInterval      massint,
	const real&             dx_finest,
	const int&              mesh_dim
);