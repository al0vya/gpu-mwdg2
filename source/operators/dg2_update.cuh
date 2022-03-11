#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_scan.cuh"

#include "BLOCK_VAR_MACROS.cuh"
#include "MortonCode.h"
#include "Neighbours.h"
#include "FlowCoeffs.h"
#include "SolverParams.h"
#include "SimulationParams.h"
#include "FlowVector.h"

#include "get_x_coord.cuh"
#include "get_y_coord.cuh"
#include "get_x_face_coord.cuh"
#include "get_y_face_coord.cuh"
#include "get_x_face_unit.cuh"
#include "get_y_face_unit.cuh"
#include "get_leg_basis.cuh"
#include "get_bed_src.cuh"
#include "flux_HLL.cuh"

__global__
void dg2_update
(
    Neighbours        d_neighbours,
    AssembledSolution d_assem_sol_load,
    AssembledSolution d_assem_sol_store,
    SolverParams      solver_params,
    SimulationParams  sim_params,
    real              dx_finest,
    real              dy_finest,
    real              dt,
    int               test_case,
    real*             d_dt_CFL,
    bool              rkdg2
);
