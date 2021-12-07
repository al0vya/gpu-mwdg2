#include "preflag_topo.cuh"

__host__
void preflag_topo
(
	ScaleCoefficients& d_scale_coeffs,
	Details&           d_details,
	bool*              d_preflagged_details,
	Maxes&             maxes, 
	SolverParams&  solver_params,
	int                first_time_step
)
{
	for (int level = solver_params.L - 1; level >= 0; level--)
	{	
		int num_threads = 1 << (2 * level);
		int num_blocks  = get_num_blocks(num_threads, THREADS_PER_BLOCK);

		encode_and_thresh_topo<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_scale_coeffs,
			d_details,
			d_preflagged_details,
			maxes,
			solver_params,
			level,
			first_time_step
		);
	}
}