#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"

#include "AssembledSolution.h"
#include "SimulationParams.h"
#include "SolverParams.h"
#include "SaveInterval.h"

#include "compact.cuh"
#include "get_lvl_idx.cuh"

typedef struct Points
{
	real ll_x;
	real ll_y;
	real ul_x;
	real ul_y;
	real lr_x;
	real lr_y;
	real ur_x;
	real ur_y;
	
} Points;

__host__ void write_soln_vtk
(
	const char*              respath,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
);