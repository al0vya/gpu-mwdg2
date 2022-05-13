#pragma once

#include "cuda_runtime.h"

#include "dilate.cuh"

__host__ __device__
MortonCode generate_morton_code
(
	Coordinate x,
	Coordinate y
);