#pragma once

#include "BLOCK_VAR_MACROS.cuh"
#include "cuda_utils.cuh"

#include "get_lvl_idx.cuh"
#include "read_and_project_modes_fv1.cuh"
#include "read_and_project_modes_dg2.cuh"

__host__
void get_modal_values
(
	NodalValues&          d_nodal_vals,
	AssembledSolution&    d_assem_sol,
	SolverParameters&     solver_params,
	SimulationParameters& sim_params,
	const int&            mesh_dim,
	const int&            interface_dim,
	const int&            test_case,
	const char*           input_filename
);