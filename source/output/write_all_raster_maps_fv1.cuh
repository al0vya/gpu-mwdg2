#pragma once

#include <cstdio>
#include <cstdlib>

#include "../utilities/cuda_utils.cuh"

#include "../classes/Points.h"
#include "../classes/AssembledSolution.h"
#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"
#include "../classes/SaveInterval.h"
#include "../classes/FinestGrid.h"

#include "write_raster_file.cuh"
#include "../zorder/get_i_index.cuh"
#include "../zorder/get_j_index.cuh"
#include "../utilities/get_lvl_idx.cuh"

void write_all_raster_maps_fv1
(
    const char*              respath,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint,
	const bool&              first_t_step
);