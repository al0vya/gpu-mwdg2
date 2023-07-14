#include "write_c_prop_data.cuh"

__host__
void write_c_prop_data
(
	const PlottingParams&    plot_params,
	const clock_t&           start,
	const SolverParams&      solver_params,
	const SimulationParams&  sim_params,
	const AssembledSolution& d_assem_sol,
	const real&              current_time,
	const real&              time_mra,
	const real&              time_solver,
	const real&              dt,
	const int&               num_cells,
	const bool&              first_t_step
)
{
	write_cumulative_data
	(
		start,
		current_time,
		time_mra,
		time_solver,
		dt,
		num_cells,
		sim_params,
		plot_params,
		first_t_step
	);

	append_reals_to_file
	(
		"qx0-c-prop",
		plot_params.dirroot,
		d_assem_sol.qx0,
		d_assem_sol.length,
		first_t_step
	);

	append_reals_to_file
	(
		"qy0-c-prop",
		plot_params.dirroot,
		d_assem_sol.qy0,
		d_assem_sol.length,
		first_t_step
	);

	if (solver_params.solver_type == MWDG2)
	{
		append_reals_to_file
		(
			"qx1x-c-prop",
			plot_params.dirroot,
			d_assem_sol.qx1x,
			d_assem_sol.length,
			first_t_step
		);

		append_reals_to_file
		(
			"qx1y-c-prop",
			plot_params.dirroot,
			d_assem_sol.qx1y,
			d_assem_sol.length,
			first_t_step
		);

		append_reals_to_file
		(
			"qy1x-c-prop",
			plot_params.dirroot,
			d_assem_sol.qy1x,
			d_assem_sol.length,
			first_t_step
		);
		append_reals_to_file
		(
			"qy1y-c-prop",
			plot_params.dirroot,
			d_assem_sol.qy1y,
			d_assem_sol.length,
			first_t_step
		);
	}
}