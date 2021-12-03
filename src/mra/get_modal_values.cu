#include "get_modal_values.cuh"

__host__
void get_modal_values
(
	NodalValues&          d_nodal_vals,
	AssembledSolution&    d_assem_sol,
	SolverParameters&     solver_params,
	SimulationParameters& sim_params,
	const int&            mesh_dim,
	const int&            interface_dim,
	const int&            test_case,
	const char*           input_filename
)
{
	if (test_case != 0) // synthetic test cases
	{
		const int num_blocks = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);

		modal_projections<<<num_blocks, THREADS_PER_BLOCK>>>
		(
			d_nodal_vals,
			d_assem_sol,
			solver_params,
			mesh_dim,
			interface_dim
		);
	}
	else // use raster information
	{		
		if (solver_params.solver_type == HWFV1)
		{
			read_and_project_modes_fv1
			(
				input_filename,
				d_assem_sol,
				mesh_dim,
				solver_params.wall_height
			);
		}
		else if (solver_params.solver_type == MWDG2)
		{
			read_and_project_modes_dg2
			(
				input_filename,
				d_assem_sol,
				d_nodal_vals,
				sim_params,
				solver_params,
				mesh_dim
			);
		}
	}
}