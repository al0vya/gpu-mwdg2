#include "decoding_kernel_single_block_hw.cuh"

__global__
void decoding_kernel_single_block_hw
(
	bool*             d_sig_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	int               level,
	int               num_threads
)
{
	for (int level_block = 0; level_block < LVL_SINGLE_BLOCK; level_block++)
	{
		int num_threads_block = 1 << (2 * level_block);
		
		int tidx = threadIdx.x; 

		if (tidx < num_threads_block)
		{
			HierarchyIndex curr_lvl_idx = get_lvl_idx(level_block);
			HierarchyIndex next_lvl_idx = get_lvl_idx(level_block + 1);

			HierarchyIndex parent = curr_lvl_idx + tidx;
			HierarchyIndex child = next_lvl_idx + 4 * tidx;

			bool is_sig = d_sig_details[parent];

			if (is_sig)
			{
				ParentScaleCoeffsHW parents  = load_parent_scale_coeffs_hw(d_scale_coeffs, parent);
				DetailHW            detail   = load_details_hw(d_details, parent);
				ChildScaleCoeffsHW  children = decode_scale_coeffs(parents, detail);

				store_scale_coeffs_vector(children, d_scale_coeffs, child);
			}
		}

		__syncthreads();
	}
}