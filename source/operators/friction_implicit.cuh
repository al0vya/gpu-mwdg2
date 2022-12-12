#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../classes/Neighbours.h"
#include "../classes/SolverParams.h"
#include "../classes/SimulationParams.h"

#include "apply_friction.cuh"
#include "../zorder/generate_morton_code.cuh"

__global__
void friction_implicit
(
	AssembledSolution    d_assem_sol,
	Neighbours           d_neighbours,
	SolverParams     solver_params, 
	SimulationParams sim_params,
	real                 dt
);