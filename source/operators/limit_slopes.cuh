#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Neighbours.h"
#include "SolverParams.h"
#include "SimulationParams.h"
#include "LegendreBasis.h"
#include "FlowCoeffs.h"

#include "get_limited_slopes.cuh"
#include "get_leg_basis.cuh"

__global__
void limit_slopes
(
    AssembledSolution d_assem_sol,
    Neighbours        d_neighbours,
    SimulationParams  sim_params,
    SolverParams      solver_params,
    real              dx_finest,
    real              dy_finest
);