#pragma once

#include "cuda_utils.cuh"

#include "AssembledSolution.h"

#include "read_raster_file.h"

__host__
void read_and_project_modes_fv1
(
	const char*              input_filename,
	const AssembledSolution& d_assem_sol,
	const int&               mesh_dim,
	const real&              wall_height
);