#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Neighbours.h"
#include "SolverParams.h"
#include "SimulationParams.h"
#include "Boundaries.h"

#include "get_lvl_idx.cuh"
#include "compact.cuh"
#include "non_reflective_wave.cuh"

__global__
void add_ghost_cells
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParams     solver_params,
	SimulationParams sim_params,
	Boundaries           boundaries,
	real                 dt,
	real                 dx_finest,
	int                  test_case
);