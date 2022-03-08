#pragma once

#include "cub/device/device_select.cuh"

#include "cuda_utils.cuh"

#include "AssembledSolution.h"
#include "Neighbours.h"
#include "CompactionFlags.h"
#include "SolverParams.h"

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