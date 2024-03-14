#include "regularisation_kernel_single_block.cuh"

__global__
void regularisation_kernel_single_block
(
	bool*          d_sig_details,
	int            level,
	int            num_threads
)
{
	__shared__ bool shared_sig_details[THREADS_PER_BLOCK];
	
	DetailChildren child_details;

	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	if (idx >= num_threads) return;

	HierarchyIndex prev_lvl_idx = get_lvl_idx(level - 1);
	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	HierarchyIndex parent_idx;
	HierarchyIndex child_idx = curr_lvl_idx + t_idx;

	shared_sig_details[t_idx] = d_sig_details[child_idx];

	__syncthreads();

	for (int lvl = LVL_SINGLE_BLOCK - 1; lvl >= 0; lvl--)
	{
		HierarchyIndex curr_lvl_idx_block = get_lvl_idx(lvl);
		int            num_threads = 1 << (2 * lvl);

		parent_idx = curr_lvl_idx_block + t_idx;

		if (t_idx < num_threads)
		{
			child_details = get_child_details
			(
				shared_sig_details,
				4 * t_idx
			);
		}

		__syncthreads();

		if (t_idx < num_threads)
		{
			if (child_details.has_sig_detail()) d_sig_details[parent_idx] = SIGNIFICANT;

			shared_sig_details[t_idx] = child_details.has_sig_detail();
		}

		__syncthreads();
	}
}