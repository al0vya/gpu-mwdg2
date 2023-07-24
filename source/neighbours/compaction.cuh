#pragma once

#include "../../cub/device/device_select.cuh"

#include "../utilities/cuda_utils.cuh"

#include "../classes/AssembledSolution.h"
#include "../classes/Neighbours.h"
#include "../classes/CompactionFlags.h"
#include "../classes/SolverParams.h"

__host__
void compaction
(
	AssembledSolution& d_buf_assem_sol, 
	AssembledSolution& d_assem_sol, 
	Neighbours&        d_buf_neighbours, 
	Neighbours&        d_neighbours, 
	CompactionFlags&   d_compaction_flags,
	int                num_finest_elems,
	const SolverParams& solver_params
);