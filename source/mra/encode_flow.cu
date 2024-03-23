#include "encode_flow.cuh"

__host__
void encode_flow
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	real*              d_norm_details,
	bool*              d_sig_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&      solver_params,
	bool               for_nghbrs
)
{
	TRACE("This is encoding.");
	
	for (int level = solver_params.L - 1; level >= LVL_SINGLE_BLOCK; level--)
	{
	    int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);
		
		if (solver_params.solver_type == HWFV1)
		{
			encode_flow_kernel_hw<<<num_blocks, THREADS_PER_BLOCK>>>
			(
				d_scale_coeffs,
				d_details,
				d_norm_details,
				d_sig_details,
				d_preflagged_details,
				maxes,
				solver_params,
				level,
				num_threads
			);
		}
		else if (solver_params.solver_type == MWDG2)
		{
			encode_flow_kernel_mw<<<num_blocks, THREADS_PER_BLOCK>>>
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

	const int level = LVL_SINGLE_BLOCK - 1;
	const int num_threads = 1 << (2 * level);
	const int num_blocks = 1;
	const int block_size = num_threads / num_blocks;

	if (solver_params.solver_type == HWFV1)
	{
		encode_flow_kernel_single_block_hw<<<num_blocks, block_size>>>
		(
			d_scale_coeffs,
			d_details,
			d_norm_details,
			d_sig_details,
			d_preflagged_details,
			maxes,
			solver_params,
			level,
			num_threads
		);
	}
	else if (solver_params.solver_type == MWDG2)
	{
		encode_flow_kernel_single_block_mw<<<num_blocks, block_size>>>
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