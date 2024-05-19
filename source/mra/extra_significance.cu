#include "extra_significance.cuh"

void extra_significance
(
	bool*         d_sig_details,
	real*         d_norm_details,
	SolverParams& solver_params
)
{	
	extra_significance_kernel_single_block<<<1, THREADS_PER_BLOCK>>>
	(
		d_sig_details, 
		d_norm_details, 
		solver_params, 
		0,
		THREADS_PER_BLOCK
	);
	
	for (int level = LVL_SINGLE_BLOCK; level < solver_params.L - 1; level++)
	{
		int  num_threads   = 1 << (2 * level);
		int  num_blocks    = get_num_blocks(num_threads, THREADS_PER_BLOCK);
		real eps_local     = solver_params.epsilon / ( 1 << (solver_params.L - level) );
		real eps_extra_sig = (solver_params.epsilon > C(0.0))
			                 ? eps_local * pow(C(2.0), M_BAR + 1)
			                 : C(9999.0);

		HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
		HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

		extra_significance_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			d_norm_details,
			eps_local,
			eps_extra_sig,
			curr_lvl_idx,
			next_lvl_idx,
			level,
			num_threads
		);
	}
}