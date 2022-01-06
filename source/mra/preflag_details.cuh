#pragma once

#include "cuda_utils.cuh"

#include "Boundaries.h"
#include "PointSources.h"
#include "GaugePoints.h"
#include "SimulationParams.h"

#include "get_lvl_idx.cuh"
#include "refine_high_wall.cuh"

__host__
bool* preflag_details
(
	const Boundaries&        boundaries,
	const PointSources&      point_sources,
	const GaugePoints&       gauge_points,
	const SimulationParams&  sim_params,
	const int&               num_details,
	const int&               max_ref_lvl
);