#include "zero_details.cuh"

void zero_details
(
	Details      d_details,
	real*        d_norm_details,
	int          num_details,
	SolverParams solver_params
)
{
	const int num_blocks = get_num_blocks(num_details, THREADS_PER_BLOCK);

	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta0.alpha, num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta0.beta,  num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta0.gamma, num_details);
	
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx0.alpha, num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx0.beta,  num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx0.gamma, num_details);
	
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy0.alpha, num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy0.beta,  num_details);
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy0.gamma, num_details);
	
	zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_norm_details, num_details);
	
	if (solver_params.solver_type == MWDG2)
	{
		zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1x.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1x.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1x.gamma, num_details);
	    
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1x.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1x.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1x.gamma, num_details);
	    
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1x.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1x.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1x.gamma, num_details);

		zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1y.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1y.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.eta1y.gamma, num_details);
	    
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1y.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1y.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qx1y.gamma, num_details);
	    
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1y.alpha, num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1y.beta,  num_details);
	    zero_array<<<num_blocks, THREADS_PER_BLOCK>>>(d_details.qy1y.gamma, num_details);
	}	
}
