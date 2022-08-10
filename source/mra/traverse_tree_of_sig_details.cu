#include "traverse_tree_of_sig_details.cuh"

__global__
void traverse_tree_of_sig_details
(
	bool*             d_sig_details,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_buf_assem_sol,
	int               num_threads,
	SolverParams      solver_params
)
{	
	__shared__ union
	{
		int            levels [4 * THREADS_PER_BLOCK];
		HierarchyIndex indices[4 * THREADS_PER_BLOCK];

	} shared;
	
	int            levels[4];
	HierarchyIndex indices[4];

	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	int block_store_step = 3 * blockIdx.x * THREADS_PER_BLOCK;

	if (idx >= num_threads) return;

	HierarchyIndex h_idx = 0;

	MortonCode curr_code = 0;
	MortonCode fine_code = 4 * idx;

	int level = 0;

	HierarchyIndex curr_lvl_idx = get_lvl_idx(level);

	bool keep_on_traversing = true;

	while (keep_on_traversing)
	{
		MortonCode curr_code = ( fine_code >> ( 2 * (solver_params.L - level) ) );

		curr_lvl_idx = get_lvl_idx(level);

		h_idx = curr_lvl_idx + curr_code;

		bool is_sig = d_sig_details[h_idx];

		if (!is_sig)
		{
			// recording z-order index and level
			for (int i = 0; i < 4; i++)
			{
				indices[i] = h_idx;
				levels[i]  = level;
			}

			keep_on_traversing = false;
		}
		else
		{
			level++;
			
			bool penultimate_level = (level == solver_params.L);

			if (!penultimate_level)
			{
				keep_on_traversing = true;
			}
			else
			{
				HierarchyIndex next_lvl_idx = get_lvl_idx(level);
				HierarchyIndex child_idx    = next_lvl_idx + 4 * curr_code;

				// recording z-order index and level
				for (int i = 0; i < 4; i++)
				{
					indices[i] = child_idx + i;
					levels[i]  = level;
				}

				keep_on_traversing = false;
			}
		}
	}
	
	// storing active indices
	for (int i = 0; i < 4; i++) shared.indices[4 * t_idx + i] = indices[i];
	__syncthreads();

	for (int i = 0; i < 4; i++) d_buf_assem_sol.act_idcs[idx + i * THREADS_PER_BLOCK + block_store_step] = shared.indices[t_idx + i * THREADS_PER_BLOCK];
	__syncthreads();

	// storing levels
	for (int i = 0; i < 4; i++) shared.levels[4 * t_idx + i] = levels[i];
	__syncthreads();

	for (int i = 0; i < 4; i++) d_buf_assem_sol.levels[idx + i * THREADS_PER_BLOCK + block_store_step] = shared.levels[t_idx + i * THREADS_PER_BLOCK];
}