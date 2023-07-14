#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/AssembledSolution.h"
#include "../classes/SolverParams.h"

#include "../input/read_raster_file.h"
#include "../input/read_keyword_str.h"

__host__
void read_and_project_modes_fv1
(
	const char*              input_filename,
	const AssembledSolution& d_assem_sol,
	const int&               mesh_dim,
	const SolverParams&      solver_params
);