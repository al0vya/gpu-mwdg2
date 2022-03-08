#include "sort_finest_scale_coeffs_z_order.cuh"

__global__
void sort_finest_scale_coeffs_z_order
(
	AssembledSolution d_buf_assem_sol,
	AssembledSolution d_assem_sol,
	MortonCode*       d_rev_z_order,
	SolverParams      solver_params
)
{
	const int idx = blockDim.x * blockIdx.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	const int sorted_idx = d_rev_z_order[idx];
	
	d_assem_sol.h0[idx]  = d_buf_assem_sol.h0[sorted_idx];
	d_assem_sol.qx0[idx] = d_buf_assem_sol.qx0[sorted_idx];
	d_assem_sol.qy0[idx] = d_buf_assem_sol.qy0[sorted_idx];
	d_assem_sol.z0[idx]  = d_buf_assem_sol.z0[sorted_idx];

	if (solver_params.solver_type == MWDG2)
	{
		d_assem_sol.h1x[idx]  = d_buf_assem_sol.h1x[sorted_idx];
	    d_assem_sol.qx1x[idx] = d_buf_assem_sol.qx1x[sorted_idx];
	    d_assem_sol.qy1x[idx] = d_buf_assem_sol.qy1x[sorted_idx];
	    d_assem_sol.z1x[idx]  = d_buf_assem_sol.z1x[sorted_idx];
		
		d_assem_sol.h1y[idx]  = d_buf_assem_sol.h1y[sorted_idx];
	    d_assem_sol.qx1y[idx] = d_buf_assem_sol.qx1y[sorted_idx];
	    d_assem_sol.qy1y[idx] = d_buf_assem_sol.qy1y[sorted_idx];
	    d_assem_sol.z1y[idx]  = d_buf_assem_sol.z1y[sorted_idx];
	}
}