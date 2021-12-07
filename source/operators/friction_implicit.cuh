#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "Neighbours.h"
#include "SolverParams.h"
#include "SimulationParams.h"

#include "apply_friction.cuh"
#include "generate_morton_code.cuh"

__global__
void friction_implicit
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParams     solver_params, 
	SimulationParams sim_params,
	real                 dt
);