#pragma once

#include <cstdio>
#include <cstring>

#include "../utilities/BLOCK_VAR_MACROS.cuh"
#include "../utilities/cuda_utils.cuh"

#include "../classes/SimulationParams.h"
#include "../classes/SaveInterval.h"

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