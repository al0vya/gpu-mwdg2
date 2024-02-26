#include "regularisation_kernel.cuh"

template <bool SINGLE_BLOCK>
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

	if (SINGLE_BLOCK)
	{
		HierarchyIndex parent_idx;
		HierarchyIndex child_idx = curr_lvl_idx + t_idx;
		
		shared_sig_details[t_idx] = d_sig_details[child_idx];

		__syncthreads();
		
		for (int lvl = LVL_SINGLE_BLOCK - 1; lvl >= 0; lvl--)
		{
			HierarchyIndex curr_lvl_idx_block = get_lvl_idx(lvl);
			int            num_threads        = 1 << (2 * lvl);

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
	else
	{
		HierarchyIndex h_idx = curr_lvl_idx + idx;

		shared_sig_details[t_idx] = d_sig_details[h_idx];

		__syncthreads();
		
		if ( t_idx >= (THREADS_PER_BLOCK / 4) ) return;

		HierarchyIndex t_idx_shifted = 4 * t_idx;
				 h_idx         = prev_lvl_idx + t_idx + blockIdx.x * (THREADS_PER_BLOCK / 4);

		child_details = get_child_details
		(
			shared_sig_details,
			t_idx_shifted
		);

		if ( child_details.has_sig_detail() ) d_sig_details[h_idx] = SIGNIFICANT;
	}	
}

inline void dummy_template_instantiator
(
	bool*    d_sig_details,
	int      level,
	int      num_threads
)
{
	regularisation_kernel<true><<<1, 1>>>
	(
		d_sig_details, 
		level,
		num_threads
	);

	regularisation_kernel<false><<<1, 1>>>
	(
		d_sig_details,
		level,
		num_threads
	);
}