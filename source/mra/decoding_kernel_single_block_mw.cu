#include "decoding_kernel_single_block_mw.cuh"

__global__
void decoding_kernel_single_block_mw
(
	bool*             d_sig_details,
	Details           d_details,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	int               level,
	int               num_threads
)
{
	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;

	for (int level_block = 0; level_block < LVL_SINGLE_BLOCK; level_block++)
	{
		int num_threads_block = 1 << (2 * level_block);

		if (idx < num_threads_block)
		{
			HierarchyIndex curr_lvl_idx = get_lvl_idx(level_block);
			HierarchyIndex next_lvl_idx = get_lvl_idx(level_block + 1);

			HierarchyIndex parent = curr_lvl_idx + idx;
			HierarchyIndex child = next_lvl_idx + 4 * idx;

			bool is_sig = d_sig_details[parent];

			if (is_sig)
			{
				ParentScaleCoeffsMW parents = load_parent_scale_coeffs_mw(d_scale_coeffs, parent);
				DetailMW            detail = load_details_mw(d_details, parent);
				ChildScaleCoeffsMW  children = decode_scale_coeffs(parents, detail);

				store_scale_coeffs(children, d_scale_coeffs, child);
			}
		}

		__syncthreads();
	}
}