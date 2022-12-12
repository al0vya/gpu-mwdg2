#pragma once

#include <cstdio>
#include <cstdlib>

#include "../utilities/cuda_utils.cuh"

#include "../classes/Points.h"
#include "../classes/AssembledSolution.h"
#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"
#include "../classes/SaveInterval.h"

#include "../zorder/compact.cuh"
#include "../utilities/get_lvl_idx.cuh"
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