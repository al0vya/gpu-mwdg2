#include "extra_significance.cuh"

template<bool SINGLE_BLOCK>
__global__
void extra_significance
(
	bool*            d_sig_details,
	real*            d_norm_details,
	SolverParameters solver_params,
	int              level,
	int              num_threads
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= num_threads) return;

	if (SINGLE_BLOCK)
	{
		for (int lvl = 0; lvl < LVL_SINGLE_BLOCK; lvl++)
		{
			HierarchyIndex curr_lvl_idx_block = get_lvl_idx(lvl);
			HierarchyIndex next_lvl_idx_block = get_lvl_idx(lvl + 1);
			int            num_threads_block  = 1 << (2 * lvl);

			if (idx < num_threads_block)
			{
				HierarchyIndex parent_idx = curr_lvl_idx_block + idx;
				HierarchyIndex child_idx  = next_lvl_idx_block + 4 * idx;

				bool is_sig = d_sig_details[parent_idx];

				real norm_detail = d_norm_details[parent_idx];

				real eps_local_block     = solver_params.epsilon / ( 1 << (solver_params.L - lvl) );
				real eps_extra_sig_block = eps_local_block * pow(C(2.0), M_BAR + 1);

				bool is_extra_sig = (norm_detail >= eps_extra_sig_block);

				if (is_sig && is_extra_sig)
				{
					d_sig_details[child_idx + 0] = SIGNIFICANT;
					d_sig_details[child_idx + 1] = SIGNIFICANT;
					d_sig_details[child_idx + 2] = SIGNIFICANT;
					d_sig_details[child_idx + 3] = SIGNIFICANT;
				}
			}

			__syncthreads();
		}
	}
	else
	{
		HierarchyIndex curr_lvl_idx = get_lvl_idx(level);
		HierarchyIndex next_lvl_idx = get_lvl_idx(level + 1);
		
		HierarchyIndex parent_idx = curr_lvl_idx + idx;

		real eps_local     = solver_params.epsilon / (1 << (solver_params.L - level));
		real eps_extra_sig = eps_local * pow(C(2.0), M_BAR + 1);

		bool sig_detail  = d_sig_details[parent_idx];
		real norm_detail = d_norm_details[parent_idx];

		bool is_extra_sig = (norm_detail >= eps_extra_sig);

		if ( !(sig_detail && is_extra_sig) ) return;

		HierarchyIndex child_idx = next_lvl_idx + 4 * idx;

		d_sig_details[child_idx + 0] = SIGNIFICANT;
		d_sig_details[child_idx + 1] = SIGNIFICANT;
		d_sig_details[child_idx + 2] = SIGNIFICANT;
		d_sig_details[child_idx + 3] = SIGNIFICANT;
	}
}

inline void dummy_template_instantiator
(
	bool*            d_sig_details,
	real*            d_norm_details,
	SolverParameters solver_params,
	int              level,
	int              num_threads
)
{
	extra_significance<false><<<1,1>>>(d_sig_details, d_norm_details, solver_params, level, num_threads);
	extra_significance<true> <<<1,1>>>(d_sig_details, d_norm_details, solver_params, level, num_threads);
}