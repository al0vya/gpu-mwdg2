#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"

#include "Points.h"
#include "AssembledSolution.h"
#include "SimulationParams.h"
#include "SolverParams.h"
#include "SaveInterval.h"

#include "compact.cuh"
#include "get_lvl_idx.cuh"
__host__ void write_soln_vtk
(
	const char*              respath,
	const AssembledSolution& d_assem_sol,
	      real*              d_dt_CFL,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
);