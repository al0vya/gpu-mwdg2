#pragma once

#include "compact.cuh"

__host__ __device__
Coordinate get_j_index(MortonCode code);