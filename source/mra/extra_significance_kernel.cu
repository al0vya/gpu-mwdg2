#include "extra_significance_kernel.cuh"

__global__
void extra_significance_kernel
(
	bool*        d_sig_details,
	real*        d_norm_details,
	SolverParams solver_params,
	int          level,
	int          num_threads
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
	HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);

	HierarchyIndex parent_idx = curr_lvl_idx + idx;

	real eps_local = solver_params.epsilon / (1 << (solver_params.L - level));
	real eps_extra_sig = eps_local * pow(C(2.0), M_BAR + 1);

	bool sig_detail = d_sig_details[parent_idx];
	real norm_detail = d_norm_details[parent_idx];

	bool is_extra_sig = (norm_detail >= eps_extra_sig);

	if (!(sig_detail && is_extra_sig)) return;

	HierarchyIndex child_idx = next_lvl_idx + 4 * idx;

	d_sig_details[child_idx + 0] = SIGNIFICANT;
	d_sig_details[child_idx + 1] = SIGNIFICANT;
	d_sig_details[child_idx + 2] = SIGNIFICANT;
	d_sig_details[child_idx + 3] = SIGNIFICANT;
}