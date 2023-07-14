#include "write_all_raster_maps.cuh"

void write_all_raster_maps
(
    const PlottingParams&    plot_params,
	const AssembledSolution& d_assem_sol,
	const real&              dx_finest,
	const real&              dy_finest,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval&      saveint,
	const bool&              first_t_step
)
{
	printf("Writing raster maps.\n");

	if (solver_params.solver_type == HWFV1)
	{
		write_all_raster_maps_fv1
		(
			plot_params,
			d_assem_sol,
			dx_finest,
			dy_finest,
			sim_params,
			solver_params,
			saveint,
			first_t_step
		);
	}
	else if (solver_params.solver_type == MWDG2)
	{
		write_all_raster_maps_dg2
		(
			plot_params,
			d_assem_sol,
			dx_finest,
			dy_finest,
			sim_params,
			solver_params,
			saveint,
			first_t_step
		);
	}
}