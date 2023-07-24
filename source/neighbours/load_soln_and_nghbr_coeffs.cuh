#pragma once

#include "device_launch_parameters.h"

#include "../../cub/block/block_radix_sort.cuh"

#include "../classes/Neighbours.h"
#include "../classes/ScaleCoefficients.h"
#include "../classes/AssembledSolution.h"
#include "../classes/SolverParams.h"

__global__
void load_soln_and_nghbr_coeffs
(
	Neighbours        d_neighbours,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_assem_sol,
	SolverParams  solver_params
);