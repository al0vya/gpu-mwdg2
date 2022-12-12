#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "../classes/Neighbours.h"
#include "../types/Coordinate.h"
#include "../types/Directions.h"
#include "../types/MortonCode.h"
#include "../classes/SimulationParams.h"

#include "../utilities/get_lvl_idx.cuh"

__global__
void find_neighbours
(
    AssembledSolution    d_assem_ol,
    Neighbours           d_neighbours,
    SimulationParams sim_params,
    int                  mesh_dim
);