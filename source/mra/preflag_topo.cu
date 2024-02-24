#include "preflag_topo.cuh"

__host__
void preflag_topo
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&      solver_params
)
{
	for (int level = solver_params.L - 1; level >= 0; level--)
	{	
		int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		if (solver_params.solver_type == HWFV1)
		{
			encode_topography_kernel_hw<<<num_blocks, THREADS_PER_BLOCK>>>
			(
				d_scale_coeffs,
				d_details,
				d_preflagged_details,
				maxes,
				solver_params,
				level
			);
		}
		else if (solver_params.solver_type == MWDG2)
		{
			encode_topography_kernel_mw<<<num_blocks, THREADS_PER_BLOCK>>>
			(
				d_scale_coeffs,
				d_details,
				d_preflagged_details,
				maxes,
				solver_params,
				level
			);
		}
	}
}