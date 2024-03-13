#include "copy_finest_coefficients.cuh"

__global__
void copy_finest_coefficients
(
	AssembledSolution d_assem_sol,
	ScaleCoefficients d_scale_coeffs,
	SolverParams      solver_params,
	HierarchyIndex    finest_lvl_idx
)
{
	HierarchyIndex idx = blockDim.x * blockIdx.x + threadIdx.x;

	if ( idx >= d_assem_sol.length ) return;

	HierarchyIndex h_idx = finest_lvl_idx + idx;

	d_scale_coeffs.eta0[h_idx] = d_assem_sol.h0[idx] + d_assem_sol.z0[idx];
	d_scale_coeffs.qx0[h_idx]  = d_assem_sol.qx0[idx];
	d_scale_coeffs.qy0[h_idx]  = d_assem_sol.qy0[idx];
	d_scale_coeffs.z0[h_idx]   = d_assem_sol.z0[idx];

	if (solver_params.solver_type == MWDG2)
	{
		d_scale_coeffs.eta1x[h_idx] = d_assem_sol.h1x[idx] + d_assem_sol.z1x[idx];
		d_scale_coeffs.qx1x[h_idx]  = d_assem_sol.qx1x[idx];
		d_scale_coeffs.qy1x[h_idx]  = d_assem_sol.qy1x[idx];
		d_scale_coeffs.z1x[h_idx]   = d_assem_sol.z1x[idx];

		d_scale_coeffs.eta1y[h_idx] = d_assem_sol.h1y[idx] + d_assem_sol.z1y[idx];
		d_scale_coeffs.qx1y[h_idx]  = d_assem_sol.qx1y[idx];
		d_scale_coeffs.qy1y[h_idx]  = d_assem_sol.qy1y[idx];
		d_scale_coeffs.z1y[h_idx]   = d_assem_sol.z1y[idx];
	}
}