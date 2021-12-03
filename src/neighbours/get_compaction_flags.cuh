#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cub/block/block_load.cuh"

#include "BLOCK_VAR_MACROS.cuh"

#include "HierarchyIndex.h"
#include "CompactionFlags.h"
#include "AssembledSolution.h"

__global__
void get_compaction_flags
(
	AssembledSolution d_assem_sol,
	CompactionFlags   d_compaction_flags,
	int               num_finest_elems
);