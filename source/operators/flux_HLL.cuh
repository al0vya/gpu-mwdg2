#pragma once

#include "cuda_runtime.h"

#include "FlowVector.h"
#include "SolverParams.h"
#include "SimulationParams.h"

__device__
FlowVector flux_HLL_x
(
	const FlowVector&       U_neg,
	const FlowVector&       U_pos,
	const SolverParams&     solver_params,
	const SimulationParams& sim_params
);

__device__
FlowVector flux_HLL_y
(
	const FlowVector&       U_neg,
	const FlowVector&       U_pos,
	const SolverParams&     solver_params,
	const SimulationParams& sim_params
);