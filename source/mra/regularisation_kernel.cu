#include "regularisation_kernel.cuh"

__global__
void regularisation_kernel
(
	bool*          d_sig_details,
	HierarchyIndex curr_lvl_idx,
	HierarchyIndex next_lvl_idx,
	int            num_threads
)
{
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	const HierarchyIndex parent_idx = curr_lvl_idx + idx;
	const HierarchyIndex child_idx  = next_lvl_idx + 4 * idx;

	SigChildren children( *reinterpret_cast<char4*>(d_sig_details + child_idx) );

	if ( children.has_sig_detail() ) d_sig_details[parent_idx] = SIGNIFICANT;
}