#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "BLOCK_VAR_MACROS.cuh"

#include "NodalValues.h"
#include "HierarchyIndex.h"
#include "Coordinate.h"
#include "AssembledSolution.h"
#include "SolverParams.h"

__global__
void modal_projections
(
	NodalValues       d_nodal_vals,
	AssembledSolution d_assem_sol,
	SolverParams  solver_params,
	int               mesh_dim,
	int               interface_dim
);