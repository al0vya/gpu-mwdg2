#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Neighbours.h"
#include "Coordinate.h"
#include "Directions.h"
#include "MortonCode.h"
#include "SimulationParams.h"

#include "get_lvl_idx.cuh"

__global__
void find_neighbours
(
    AssembledSolution    d_assem_ol,
    Neighbours           d_neighbours,
    SimulationParams sim_params,
    int                  mesh_dim
);