#include "traverse_tree_of_sig_details.cuh"

__global__
void traverse_tree_of_sig_details
(
	bool*             d_sig_details,
	ScaleCoefficients d_scale_coeffs,
	AssembledSolution d_buf_assem_sol,
	int               num_threads,
	SolverParameters  solver_params
)
{	
	__shared__ union
	{
		real           coeffs [4 * THREADS_PER_BLOCK];
		int            levels [4 * THREADS_PER_BLOCK];
		HierarchyIndex indices[4 * THREADS_PER_BLOCK];

	} shared;
	
	real     h_arr[4];
	real     qx_arr[4];
	real     qy_arr[4];
	real     z_arr[4];
	int      levels[4];
	
	HierarchyIndex indices[4];

	HierarchyIndex t_idx = threadIdx.x;
	HierarchyIndex idx   = blockIdx.x * blockDim.x + t_idx;

	int block_store_step = 3 * blockIdx.x * THREADS_PER_BLOCK;

	if (idx >= num_threads) return;

	HierarchyIndex  stack[52];
	HierarchyIndex *stackPtr = stack;
	*stackPtr++ = NULL;

	HierarchyIndex h_idx = 0;

	MortonCode fine_code = 4 * idx;

	int level = 0;

	do
	{
		HierarchyIndex curr_lvl_idx = get_lvl_idx(level);

		HierarchyIndex local_idx = h_idx - curr_lvl_idx;
		
		MortonCode current_code = local_idx;

		bool is_child = ( ( fine_code >> ( 2 * (solver_params.L - level) ) ) == current_code);

		if (is_child)
		{
			bool is_sig = d_sig_details[h_idx];

			if (!is_sig)
			{				
				real eta = d_scale_coeffs.eta0[h_idx];
				real qx  = d_scale_coeffs.qx0[h_idx];
				real qy  = d_scale_coeffs.qy0[h_idx];
				real z   = d_scale_coeffs.z0[h_idx];

				#pragma unroll
				for (int i = 0; i < 4; i++)
				{
					h_arr[i]   = eta - z;
					qx_arr[i]  = qx;
					qy_arr[i]  = qy;
					z_arr[i]   = z;
					indices[i] = h_idx;
					levels[i]  = level;
				}

				goto store;
			}
			else
			{
				bool penultimate_level = (++level == solver_params.L);
				
				HierarchyIndex next_lvl_idx = get_lvl_idx(level);
				
				HierarchyIndex child_idx = next_lvl_idx + 4 * local_idx;

				if (!penultimate_level)
				{
					// get child indices and make index child_0 of current sub-element
					h_idx       = child_idx + 0;
					*stackPtr++ = child_idx + 1;
					*stackPtr++ = child_idx + 2;
					*stackPtr++ = child_idx + 3;
				}
				else
				{
					// reached penultimate level, add information to last level and exit
					
					#pragma unroll
					for (int i = 0; i < 4; i++)
					{
						// disrupted ordering of coefficients to ensure cache hit for z, as z is used twice
						// once to calculate h, and once when simply storing z
						real eta = d_scale_coeffs.eta0[child_idx + i];
						real z   = d_scale_coeffs.z0[child_idx + i];

						h_arr[i]   = eta - z;
						z_arr[i]   = z;
						qx_arr[i]  = d_scale_coeffs.qx0[child_idx + i];
						qy_arr[i]  = d_scale_coeffs.qy0[child_idx + i];
						indices[i] = child_idx + i;
						levels[i]  = level;
					}
					
					goto store;
				}
			}
		}
		else
		{
			h_idx = *--stackPtr;
		}
	}
	while (NULL != h_idx);

	store:
	{
		// storing active indices
		#pragma unroll
		for (int i = 0; i < 4; i++) shared.indices[4 * t_idx + i] = indices[i];
		__syncthreads();
		#pragma unroll
		for (int i = 0; i < 4; i++) d_buf_assem_sol.act_idcs[idx + i * THREADS_PER_BLOCK + block_store_step] = shared.indices[t_idx + i * THREADS_PER_BLOCK];
		__syncthreads();
		
		// storing levels
		#pragma unroll
		for (int i = 0; i < 4; i++) shared.levels[4 * t_idx + i] = levels[i];
		__syncthreads();
		#pragma unroll
		for (int i = 0; i < 4; i++) d_buf_assem_sol.levels[idx + i * THREADS_PER_BLOCK + block_store_step] = shared.levels[t_idx + i * THREADS_PER_BLOCK];
	}
}