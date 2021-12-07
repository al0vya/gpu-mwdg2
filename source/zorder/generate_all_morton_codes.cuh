#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MortonCode.h"
#include "HierarchyIndex.h"

#include "generate_morton_code.cuh"

__global__
void generate_all_morton_codes
(
	MortonCode* d_morton_codes,
	int*        d_indices,
	int         mesh_dim
);