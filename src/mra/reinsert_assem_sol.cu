#include "reinsert_assem_sol.cuh"

__global__
void reinsert_assem_sol
(
	AssembledSolution d_assem_sol,
	HierarchyIndex*   act_idcs,
	ScaleCoefficients d_scale_coeffs,
	SolverParams  solver_params
)
{
	HierarchyIndex idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= d_assem_sol.length) return;

	HierarchyIndex active_idx = act_idcs[idx];

	d_scale_coeffs.eta0[active_idx] = d_assem_sol.h0[idx] + d_assem_sol.z0[idx];
	d_scale_coeffs.qx0[active_idx]  = d_assem_sol.qx0[idx];
	d_scale_coeffs.qy0[active_idx]  = d_assem_sol.qy0[idx];
	
	if (solver_params.solver_type == MWDG2)
	{
		d_scale_coeffs.eta1x[active_idx] = d_assem_sol.h1x[idx] + d_assem_sol.z1x[idx];
		d_scale_coeffs.qx1x[active_idx]  = d_assem_sol.qx1x[idx];
		d_scale_coeffs.qy1x[active_idx]  = d_assem_sol.qy1x[idx];
		
		d_scale_coeffs.eta1y[active_idx] = d_assem_sol.h1y[idx] + d_assem_sol.z1y[idx];
		d_scale_coeffs.qx1y[active_idx]  = d_assem_sol.qx1y[idx];
		d_scale_coeffs.qy1y[active_idx]  = d_assem_sol.qy1y[idx];
	}
}