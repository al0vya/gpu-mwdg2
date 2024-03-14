#include "regularisation_kernel.cuh"

__global__
void regularisation_kernel
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

	HierarchyIndex h_idx = curr_lvl_idx + idx;

	shared_sig_details[t_idx] = d_sig_details[h_idx];

	__syncthreads();

	if (t_idx >= (THREADS_PER_BLOCK / 4)) return;

	HierarchyIndex t_idx_shifted = 4 * t_idx;
	h_idx = prev_lvl_idx + t_idx + blockIdx.x * (THREADS_PER_BLOCK / 4);

	child_details = get_child_details
	(
		shared_sig_details,
		t_idx_shifted
	);

	if (child_details.has_sig_detail()) d_sig_details[h_idx] = SIGNIFICANT;
}