#pragma once

#include <cstdio>
#include <cstdlib>

#include "cuda_utils.cuh"

#include "FlowCoeffs.h"
#include "Points.h"
#include "AssembledSolution.h"
#include "SimulationParams.h"
#include "SolverParams.h"
#include "SaveInterval.h"
#include "FinestGrid.h"

#include "write_reals_to_file.cuh"
#include "get_i_index.cuh"
#include "get_j_index.cuh"
#include "get_lvl_idx.cuh"

void write_soln_planar_dg2
(
    const char*              respath,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint
);