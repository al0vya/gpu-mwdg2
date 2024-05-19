#include "extra_significance_kernel.cuh"

__global__
void extra_significance_kernel
(
	bool*          d_sig_details,
	real*          d_norm_details,
	real           eps_local,
	real           eps_extra_sig,
	HierarchyIndex curr_lvl_idx,
	HierarchyIndex next_lvl_idx,
	int            level,
	int            num_threads
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	HierarchyIndex parent_idx   = curr_lvl_idx + idx;
	HierarchyIndex child_idx    = next_lvl_idx + 4 * idx;

	bool sig_detail  = d_sig_details[parent_idx];
	real norm_detail = d_norm_details[parent_idx];

	bool is_extra_sig = (norm_detail >= eps_extra_sig);

	if ( (sig_detail && is_extra_sig) )
	{
		reinterpret_cast<char4*>(d_sig_details + child_idx)[0] =
		{
			SIGNIFICANT,
			SIGNIFICANT,
			SIGNIFICANT,
			SIGNIFICANT
		};
	}

	
}