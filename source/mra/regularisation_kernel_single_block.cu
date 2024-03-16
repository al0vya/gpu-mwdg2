#include "regularisation_kernel_single_block.cuh"

__global__
void regularisation_kernel_single_block
(
	bool* d_sig_details
)
{
	for (int lvl = LVL_SINGLE_BLOCK; lvl >= 0; lvl--)
	{
		int idx = threadIdx.x;
		int num_threads_active = 1 << (2 * lvl);

		if (idx < num_threads_active)
		{
			HierarchyIndex curr_lvl_idx = get_lvl_idx(lvl);
			HierarchyIndex next_lvl_idx = get_lvl_idx(lvl + 1);

			HierarchyIndex parent_idx = curr_lvl_idx + idx;
			HierarchyIndex child_idx  = next_lvl_idx + 4 * idx;

			SigChildren children( *reinterpret_cast<char4*>(d_sig_details + child_idx) );

			if ( children.has_sig_detail() ) d_sig_details[parent_idx] = SIGNIFICANT;
		}

		__syncthreads();
	}
}