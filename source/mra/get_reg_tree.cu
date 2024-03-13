#include "get_reg_tree.cuh"

__host__
void get_reg_tree
(
	bool*        d_sig_details,
	SolverParams solver_params
)
{
	for (int level = solver_params.L - 1; level > LVL_SINGLE_BLOCK; level--)
	{		
		int num_threads  = 1 << (2 * level);
		int num_blocks = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		regularisation<false><<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			level,
			num_threads
		);
	}

	HierarchyIndex prev_lvl_idx = get_lvl_idx(LVL_SINGLE_BLOCK - 1);
	HierarchyIndex curr_lvl_idx = get_lvl_idx(LVL_SINGLE_BLOCK);
	HierarchyIndex next_lvl_idx = get_lvl_idx(LVL_SINGLE_BLOCK + 1);
    
	int num_threads  = 1 << (2 * LVL_SINGLE_BLOCK);

	regularisation<true><<<1, THREADS_PER_BLOCK>>>
	(
		d_sig_details,
		LVL_SINGLE_BLOCK,
		num_threads
	);
}