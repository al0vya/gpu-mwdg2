#include "decoding.cuh"

void decoding
(
	bool*              d_sig_details,
	real*              d_norm_details,
	Details&           d_details,
	ScaleCoefficients& d_scale_coeffs,
	SolverParams&      solver_params
)
{	
	if (solver_params.solver_type == HWFV1)
	{
		decoding_kernel_single_block_hw<<<1, THREADS_PER_BLOCK>>>
		(
			d_sig_details, 
			d_details, 
			d_scale_coeffs, 
			solver_params, 
			0, 
			THREADS_PER_BLOCK
		);
	}
	else if (solver_params.solver_type == MWDG2)
	{
		decoding_kernel_single_block_mw<<<1, THREADS_PER_BLOCK>>>
		(
			d_sig_details, 
			d_details, 
			d_scale_coeffs, 
			solver_params, 
			0, 
			THREADS_PER_BLOCK
		);
	}

	for (int level = LVL_SINGLE_BLOCK; level < solver_params.L; level++)
	{				
		int            num_threads  = 1 << (2 * level);
		int            num_blocks   = get_num_blocks(num_threads, THREADS_PER_BLOCK);
		HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
		HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

		if (solver_params.solver_type == HWFV1)
		{
			decoding_kernel_hw<<<num_blocks, THREADS_PER_BLOCK>>>
			(
				d_sig_details, 
				d_details, 
				d_scale_coeffs, 
				solver_params, 
				curr_lvl_idx,
				next_lvl_idx,
				num_threads
			);
		}
		else if (solver_params.solver_type == MWDG2)
		{
			decoding_kernel_mw<<<num_blocks, THREADS_PER_BLOCK>>>
			(
				d_sig_details, 
				d_details, 
				d_scale_coeffs, 
				solver_params, 
				curr_lvl_idx,
				next_lvl_idx,
				num_threads
			);
		}
	}
}