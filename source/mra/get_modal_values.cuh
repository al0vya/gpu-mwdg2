#pragma once

#include "../utilities/BLOCK_VAR_MACROS.cuh"
#include "../utilities/cuda_utils.cuh"

#include "../utilities/get_lvl_idx.cuh"
#include "../input/read_and_project_modes_fv1.cuh"
#include "../input/read_and_project_modes_dg2.cuh"

__host__
void get_modal_values
(
	NodalValues&          d_nodal_vals,
	AssembledSolution&    d_assem_sol,
	SolverParams&     solver_params,
	SimulationParams& sim_params,
	const int&            mesh_dim,
	const int&            interface_dim,
	const int&            test_case,
	const char*           input_filename
);