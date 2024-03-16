#include "extra_significance_kernel_single_block.cuh"

__global__
void extra_significance_kernel_single_block
(
	bool*        d_sig_details,
	real*        d_norm_details,
	SolverParams solver_params,
	int          level,
	int          num_threads
)
{
	for (int lvl = 0; lvl < LVL_SINGLE_BLOCK; lvl++)
	{
		int idx = blockIdx.x * blockDim.x + threadIdx.x; 
		int num_threads_block = 1 << (2 * lvl);

		if (idx < num_threads_block)
		{
			HierarchyIndex curr_lvl_idx = get_lvl_idx(lvl);
	        HierarchyIndex next_lvl_idx = get_lvl_idx(lvl + 1);
	        HierarchyIndex parent_idx   = curr_lvl_idx + idx;
	        HierarchyIndex child_idx    = next_lvl_idx + 4 * idx;
	        
	        real eps_local     = solver_params.epsilon / (1 << (solver_params.L - lvl));
	        real eps_extra_sig = eps_local * pow(C(2.0), M_BAR + 1);
	        
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

		__syncthreads();
	}
}