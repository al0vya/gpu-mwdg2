#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "Neighbours.h"
#include "SolverParams.h"
#include "SimulationParams.h"
#include "FlowVector.h"

#include "flux_HLL.cuh"
#include "get_bed_src.cuh"

__global__
void fv1_update
(
    Neighbours           d_neighbours,
    AssembledSolution    d_assem_sol,
    SolverParams     solver_params,
    SimulationParams sim_params,
    real                 dx_finest,
    real                 dy_finest,
    real                 dt,
    real*                d_dt_CFL
);
