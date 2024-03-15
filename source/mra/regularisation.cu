#include "regularisation.cuh"

__host__
void regularisation
(
	bool*        d_sig_details,
	SolverParams solver_params
)
{
	for (int level = solver_params.L - 1; level > LVL_SINGLE_BLOCK; level--)
	{		
		int num_threads  = 1 << (2 * level);
		int num_blocks = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		regularisation_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			level,
			num_threads
		);
	}

	int num_threads  = 1 << (2 * LVL_SINGLE_BLOCK);

	regularisation_kernel_single_block<<<1, THREADS_PER_BLOCK>>>
	(
		d_sig_details,
		LVL_SINGLE_BLOCK,
		num_threads
	);
}