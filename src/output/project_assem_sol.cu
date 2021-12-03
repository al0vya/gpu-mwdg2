#include "project_assem_sol.cuh"

__host__
void project_assem_sol
(
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParameters&  solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol
)
{
	int num_blocks_sol = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);

	reinsert_assem_sol<<<num_blocks_sol, THREADS_PER_BLOCK>>>
	(
		d_assem_sol,
		d_assem_sol.act_idcs,
		d_scale_coeffs,
		solver_params
	);

	int num_finest_elems      = mesh_dim * mesh_dim;
	int num_threads_traversal = num_finest_elems / 4;
	int num_blocks_traversal  = get_num_blocks(num_threads_traversal, THREADS_PER_BLOCK);
	
	d_plot_assem_sol.length = num_finest_elems;

	traverse_tree_of_sig_details<<<num_blocks_traversal, THREADS_PER_BLOCK>>>
	(
		d_sig_details, 
		d_scale_coeffs,
		d_buf_assem_sol,
		num_threads_traversal,
		solver_params
	);

	load_soln_coeffs<<<get_num_blocks(num_finest_elems, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>
	(
		d_buf_assem_sol,
		d_scale_coeffs,
		num_finest_elems
	);

	rev_z_order_assem_sol
	(
		d_rev_z_order,
		d_indices,
		d_buf_assem_sol,
		d_plot_assem_sol,
		num_finest_elems
	);
}

__global__
void load_soln_coeffs
(
	AssembledSolution d_assem_sol,
	ScaleCoefficients d_scale_coeffs,
	int               num_finest_elems
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= num_finest_elems) return;

	HierarchyIndex act_idx = d_assem_sol.act_idcs[idx];

	d_assem_sol.h0[idx]  = d_scale_coeffs.eta0[act_idx] - d_scale_coeffs.z0[act_idx];
	d_assem_sol.qx0[idx] = d_scale_coeffs.qx0[act_idx];
	d_assem_sol.qy0[idx] = d_scale_coeffs.qy0[act_idx];
	d_assem_sol.z0[idx]  = d_scale_coeffs.z0[act_idx];
}