#pragma once

#include "compact.cuh"

__host__ __device__
Coordinate get_i_index(MortonCode code);