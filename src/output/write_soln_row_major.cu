#include "write_soln_row_major.cuh"

__host__
void write_soln_row_major
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParameters&  solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol
)
{
	project_assem_sol
	(
		mesh_dim,
		d_sig_details,
		d_scale_coeffs,
		d_buf_assem_sol,
		solver_params,
		d_rev_z_order,
		d_indices,
		d_assem_sol,
		d_plot_assem_sol
	); 
	
	write_reals_to_file("depths",      respath, d_plot_assem_sol.h0,  d_plot_assem_sol.length);
	write_reals_to_file("discharge_x", respath, d_plot_assem_sol.qx0, d_plot_assem_sol.length);
	write_reals_to_file("discharge_y", respath, d_plot_assem_sol.qy0, d_plot_assem_sol.length);
	write_reals_to_file("topo",        respath, d_plot_assem_sol.z0,  d_plot_assem_sol.length);
}