#include "write_c_prop_data.cuh"

__host__
void write_c_prop_data
(
	const char*              respath,
	const clock_t&           start,
	const SolverParams&      solver_params,
	const AssembledSolution& d_assem_sol,
	const real&              time_now,
	const real&              dt,
	const int&               num_cells,
	const bool&              first_t_step
)
{
	write_cumulative_data
	(
		start,
		time_now,
		dt,
		num_cells,
		respath,
		first_t_step
	);

	append_reals_to_file
	(
		"qx0-c-prop",
		respath,
		d_assem_sol.qx0,
		d_assem_sol.length,
		first_t_step
	);

	append_reals_to_file
	(
		"qy0-c-prop",
		respath,
		d_assem_sol.qy0,
		d_assem_sol.length,
		first_t_step
	);

	if (solver_params.solver_type == MWDG2)
	{
		append_reals_to_file
		(
			"qx1x-c-prop",
			respath,
			d_assem_sol.qx1x,
			d_assem_sol.length,
			first_t_step
		);

		append_reals_to_file
		(
			"qx1y-c-prop",
			respath,
			d_assem_sol.qx1y,
			d_assem_sol.length,
			first_t_step
		);

		append_reals_to_file
		(
			"qy1x-c-prop",
			respath,
			d_assem_sol.qy1x,
			d_assem_sol.length,
			first_t_step
		);
		append_reals_to_file
		(
			"qy1y-c-prop",
			respath,
			d_assem_sol.qy1y,
			d_assem_sol.length,
			first_t_step
		);
	}
}