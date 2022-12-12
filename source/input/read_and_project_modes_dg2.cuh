#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/SimulationParams.h"

#include "../input/read_raster_file.h"
#include "../utilities/get_num_blocks.h"

#include "../mra/modal_projections.cuh"

__host__
void read_and_project_modes_dg2
(
	const char*                 input_filename,
	const AssembledSolution&    d_assem_sol,
	const NodalValues&          d_nodal_vals,
	const SimulationParams& sim_params,
	const SolverParams&     solver_params,
	const int&                  mesh_dim
);