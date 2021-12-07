#pragma once

#include "AssembledSolution.h"
#include "SolverParams.h"

#include "write_raster_file.cuh"

__host__
void write_all_raster_maps
(
	const char*                 respath,
	const AssembledSolution&    d_assem_sol,
	const SimulationParams& sim_params,
	const SolverParams&     solver_params,
	const SaveInterval          massint,
	const int&                  mesh_dim,
	const real&                 dx_finest,
	const bool                  first_t_step
);