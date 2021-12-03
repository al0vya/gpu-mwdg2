#include "encoding_all.cuh"

__host__
void encoding_all
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	real*              d_norm_details,
	bool*              d_sig_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParameters&  solver_params,
	bool               for_nghbrs
)
{
	for (int level = solver_params.L - 1; level >= LVL_SINGLE_BLOCK; level--)
	{
	    int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		encode_and_thresh_flow<false><<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_scale_coeffs,
			d_details,
			d_norm_details,
			d_sig_details,
			d_preflagged_details,
			maxes,
			solver_params,
			level,
			num_threads,
			for_nghbrs
		);
	}

	for (int level = LVL_SINGLE_BLOCK - 1; level >= 0; level--)
	{
		int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		encode_and_thresh_flow<true><<<1, THREADS_PER_BLOCK>>>
		(
			d_scale_coeffs,
			d_details,
			d_norm_details,
			d_sig_details,
			d_preflagged_details,
			maxes,
			solver_params,
			level,
			num_threads,
			for_nghbrs
		);
	}
}