#include "write_soln_row_major.cuh"

__host__
void write_soln_row_major
(
	const char*              respath,
	const int&               mesh_dim,
	bool*&                   d_sig_details,
	const ScaleCoefficients& d_scale_coeffs,
	AssembledSolution        d_buf_assem_sol,
	const SolverParams&      solver_params,
	MortonCode*              d_rev_z_order,
	MortonCode*              d_indices,
	AssembledSolution        d_assem_sol,
	AssembledSolution        d_plot_assem_sol,
	SaveInterval&            saveint
)
{
	char depths[64]      = {'\0'};
	char discharge_x[64] = {'\0'};
	char discharge_y[64] = {'\0'};
	char topo[64]        = {'\0'}; 
	
	sprintf(depths,      "%s%d", "depths-",      saveint.count - 1);
	sprintf(discharge_x, "%s%d", "discharge_x-", saveint.count - 1);
	sprintf(discharge_y, "%s%d", "discharge_y-", saveint.count - 1);
	sprintf(topo,        "%s%d", "topo-",        saveint.count - 1);

	write_reals_to_file(depths,      respath, d_plot_assem_sol.h0,  d_plot_assem_sol.length);
	write_reals_to_file(discharge_x, respath, d_plot_assem_sol.qx0, d_plot_assem_sol.length);
	write_reals_to_file(discharge_y, respath, d_plot_assem_sol.qy0, d_plot_assem_sol.length);
	write_reals_to_file(topo,        respath, d_plot_assem_sol.z0,  d_plot_assem_sol.length);
}