#include "regularisation.cuh"

__host__
void regularisation
(
	bool*        d_sig_details,
	SolverParams solver_params
)
{
	for (int level = solver_params.L - 2; level > LVL_SINGLE_BLOCK; level--)
	{		
		int num_threads             = 1 << (2 * level);
		int num_blocks              = get_num_blocks(num_threads, THREADS_PER_BLOCK);
		HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
		HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

		regularisation_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			curr_lvl_idx,
			next_lvl_idx,
			num_threads
		);
	}

	regularisation_kernel_single_block<<<1, THREADS_PER_BLOCK>>>
	(
		d_sig_details
	);
}