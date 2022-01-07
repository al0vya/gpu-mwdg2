#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"

#include "Points.h"
#include "AssembledSolution.h"
#include "SimulationParams.h"
#include "SolverParams.h"
#include "SaveInterval.h"
#include "FinestGrid.h"

#include "compact.cuh"
#include "get_lvl_idx.cuh"

void write_soln_planar_fv1
(
    const char*              respath,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
);