#include "get_max_scale_coeffs.cuh"

__host__
Maxes get_max_scale_coeffs
(
	AssembledSolution& d_assem_sol,
	real*&             d_eta_temp
)
{
	Maxes maxes = { C(1.0), C(1.0), C(1.0), C(1.0), C(1.0) };
	
	int num_blocks = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);
	
	init_eta_temp<<<num_blocks, THREADS_PER_BLOCK>>>
	(
		d_assem_sol, 
		d_eta_temp
	);

	maxes.eta = max( C(1.0), get_max_from_array(d_eta_temp,      d_assem_sol.length) );
	maxes.h   = max( C(1.0), get_max_from_array(d_assem_sol.h0,  d_assem_sol.length) );
	maxes.qx  = max( C(1.0), get_max_from_array(d_assem_sol.qx0, d_assem_sol.length) );
	maxes.qy  = max( C(1.0), get_max_from_array(d_assem_sol.qx0, d_assem_sol.length) );
	maxes.z   = max( C(1.0), get_max_from_array(d_assem_sol.z0,  d_assem_sol.length) );

	return maxes;
}