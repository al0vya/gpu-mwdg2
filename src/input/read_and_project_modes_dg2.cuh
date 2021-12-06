#pragma once

#include "cuda_utils.cuh"

#include "SimulationParams.h"

#include "read_raster_file.h"
#include "get_num_blocks.h"

#include "modal_projections.cuh"

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