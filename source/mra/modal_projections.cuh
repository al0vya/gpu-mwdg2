#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../utilities/BLOCK_VAR_MACROS.cuh"

#include "../classes/NodalValues.h"
#include "../types/HierarchyIndex.h"
#include "../types/Coordinate.h"
#include "../classes/AssembledSolution.h"
#include "../classes/SolverParams.h"

__global__
void modal_projections
(
	NodalValues       d_nodal_vals,
	AssembledSolution d_assem_sol,
	SolverParams  solver_params,
	int               mesh_dim,
	int               interface_dim
);