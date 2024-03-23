#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../classes/Boundaries.h"
#include "../classes/PointSources.h"
#include "../classes/StagePoints.h"
#include "../classes/SimulationParams.h"
#include "../classes/SolverParams.h"

#include "../utilities/get_lvl_idx.cuh"
#include "refine_high_wall.cuh"

__host__
bool* preflag_details
(
	const Boundaries&        boundaries,
	const PointSources&      point_sources,
	const StagePoints&       stage_points,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const int&               max_ref_lvl,
	const int&               test_case
);