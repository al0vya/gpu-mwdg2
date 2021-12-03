#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Neighbours.h"
#include "SolverParameters.h"
#include "SimulationParameters.h"
#include "Boundaries.h"

#include "get_lvl_idx.cuh"
#include "compact.cuh"
#include "non_reflective_wave.cuh"

__global__
void add_ghost_cells
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParameters     solver_params,
	SimulationParameters sim_params,
	Boundaries           boundaries,
	real                 dt,
	real                 dx_finest,
	int                  test_case
);