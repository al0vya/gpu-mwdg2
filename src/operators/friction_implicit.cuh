#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "Neighbours.h"
#include "SolverParameters.h"
#include "SimulationParameters.h"

#include "apply_friction.cuh"
#include "generate_morton_code.cuh"

__global__
void friction_implicit
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParameters     solver_params, 
	SimulationParameters sim_params,
	real                 dt
);