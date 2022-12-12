#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../classes/Neighbours.h"
#include "../classes/SolverParams.h"
#include "../classes/SimulationParams.h"
#include "../classes/Boundaries.h"

#include "../utilities/get_lvl_idx.cuh"
#include "../zorder/compact.cuh"
#include "../operators/non_reflective_wave.cuh"

__global__
void add_ghost_cells
(
	AssembledSolution d_assem_sol,
	Neighbours        d_neighbours,
	SolverParams      solver_params,
	SimulationParams  sim_params,
	Boundaries        boundaries,
	real              current_time,
	real              dt,
	real              dx_finest,
	int               test_case
);