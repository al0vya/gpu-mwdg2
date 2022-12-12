#pragma once

#include "../utilities/get_num_blocks.h"

#include "../mra/reinsert_assem_sol.cuh"
#include "../mra/traverse_tree_of_sig_details.cuh"
#include "../zorder/rev_z_order_assem_sol.cuh"
#include "write_reals_to_file.cuh"
#include "write_int_to_file.cuh"

__global__
void load_soln_coeffs
(
	AssembledSolution d_assem_sol,
	ScaleCoefficients d_scale_coeffs,
	int               num_finest_elems
);

__host__
void project_assem_sol
(
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParams&  solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol
);