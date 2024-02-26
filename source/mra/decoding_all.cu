#include "decoding_all.cuh"

void decoding_all
(
	bool*              d_sig_details,
	real*              d_norm_details,
	Details&           d_details,
	ScaleCoefficients& d_scale_coeffs,
	SolverParams&      solver_params
)
{	
	extra_significance<true><<<1, THREADS_PER_BLOCK>>>
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

		extra_significance<false><<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details,
			d_norm_details,
			solver_params,
			level,
			num_threads
		);
	}
	
	decoding_kernel_single_block<<<1, THREADS_PER_BLOCK>>>
	(
		d_sig_details, 
		d_details, 
		d_scale_coeffs, 
		solver_params, 
		0, 
		THREADS_PER_BLOCK
	);

	for (int level = LVL_SINGLE_BLOCK; level < solver_params.L; level++)
	{				
		int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		decoding_kernel<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_sig_details, 
			d_details, 
			d_scale_coeffs, 
			solver_params, 
			level, 
			num_threads
		);
	}
}