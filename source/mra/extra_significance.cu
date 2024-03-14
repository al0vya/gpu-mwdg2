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
		int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		extra_significance_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			d_norm_details,
			solver_params,
			level,
			num_threads
		);
	}
}