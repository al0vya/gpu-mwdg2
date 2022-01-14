#include "write_all_raster_maps.cuh"

__host__
void write_all_raster_maps
(
	const char*              respath,
	const AssembledSolution& d_assem_sol,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const SaveInterval       massint,
	const int&               mesh_dim,
	const real&              dx_finest,
	const bool               first_t_step
)
{
	printf("Writing raster file.\n");
	
	if (first_t_step)
	{
		write_raster_file
		(
			respath,
			"dem",
			d_assem_sol.z0,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		if (solver_params.solver_type == MWDG2)
		{
			/*write_raster_file
			(
				respath,
				"dem1x",
				d_assem_sol.z1x,
				sim_params,
				massint,
				dx_finest,
				mesh_dim
			);

			write_raster_file
			(
				respath,
				"dem1y",
				d_assem_sol.z1y,
				sim_params,
				massint,
				dx_finest,
				mesh_dim
			);*/
		}
	}

	write_raster_file
	(
		respath,
		"wd",
		d_assem_sol.h0,
		sim_params,
		massint,
		dx_finest,
		mesh_dim
	);

	write_raster_file
	(
		respath,
		"qx",
		d_assem_sol.qx0,
		sim_params,
		massint,
		dx_finest,
		mesh_dim
	);

	write_raster_file
	(
		respath,
		"qy",
		d_assem_sol.qy0,
		sim_params,
		massint,
		dx_finest,
		mesh_dim
	);

	if (solver_params.solver_type == MWDG2)
	{
		/*write_raster_file
		(
			respath,
			"wd1x",
			d_assem_sol.h1x,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			respath,
			"wd1y",
			d_assem_sol.h1y,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			respath,
			"qx1x",
			d_assem_sol.qx1x,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			respath,
			"qx1y",
			d_assem_sol.qx1y,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			respath,
			"qy1x",
			d_assem_sol.qy1x,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);

		write_raster_file
		(
			respath,
			"qy1y",
			d_assem_sol.qy1y,
			sim_params,
			massint,
			dx_finest,
			mesh_dim
		);*/
	}
}