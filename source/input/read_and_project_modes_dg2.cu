#include "../input/read_and_project_modes_dg2.cuh"

__host__
void read_and_project_modes_dg2
(
	const char*              input_filename,
	const AssembledSolution& d_assem_sol,
	const NodalValues&       d_nodal_vals,
	const SimulationParams&  sim_params,
	const SolverParams&      solver_params,
	const int&               mesh_dim
)
{
	char h_raster_filename_buf[128]  = {'\0'};
	char qx_raster_filename_buf[128] = {'\0'};
	char qy_raster_filename_buf[128] = {'\0'};
	char dem_filename_buf[128]       = {'\0'};

	read_keyword_str(input_filename, "DEMfile",   7, dem_filename_buf);
	
	// buffer for DEM filename should never be null char because DEM is always needed for realistic test case
	if (dem_filename_buf[0] == '\0')
	{
		fprintf(stderr, "Error reading DEM filename, file: %s, line: %d.\n", __FILE__, __LINE__);
	}
	
	read_keyword_str(input_filename, "startfile", 9, h_raster_filename_buf);

	real* h_raster  = new real[d_assem_sol.length]();
	real* qx_raster = new real[d_assem_sol.length]();
	real* qy_raster = new real[d_assem_sol.length]();
	real* dem       = new real[d_assem_sol.length]();

	for (int i = 0; i < d_assem_sol.length; i++)
	{
		dem[i] = solver_params.wall_height;
	}

	read_raster_file(h_raster_filename_buf,  h_raster,  mesh_dim, solver_params.wall_height);
	read_raster_file(dem_filename_buf,       dem,       mesh_dim, solver_params.wall_height);

	if (solver_params.startq2d)
	{
		sprintf(qx_raster_filename_buf, "%s%s", h_raster_filename_buf, ".Qx");
		sprintf(qy_raster_filename_buf, "%s%s", h_raster_filename_buf, ".Qy");

		read_raster_file(qx_raster_filename_buf, qx_raster, mesh_dim, solver_params.wall_height);
		read_raster_file(qy_raster_filename_buf, qy_raster, mesh_dim, solver_params.wall_height);
	}

	const int interface_dim = mesh_dim + 1;
	const int interface_y   = sim_params.ysz + 1;
	const int interface_x   = sim_params.xsz + 1;
	
	real* h_interface   = new real[interface_dim * interface_dim]();
	real* qx_interface  = new real[interface_dim * interface_dim]();
	real* qy_interface  = new real[interface_dim * interface_dim]();
	real* dem_interface = new real[interface_dim * interface_dim]();

	for (Coordinate y = 0; y < interface_dim; y++)
	{
		for (Coordinate x = 0; x < interface_dim; x++)
		{
			bool wall_or_border_x = (x == interface_x - 1 || x == interface_dim - 1);
			bool wall_or_border_y = (y == interface_y - 1 || y == interface_dim - 1);

			int i = (wall_or_border_x) ? x - 1 : x;
			int j = (wall_or_border_y) ? y - 1 : y;

			int interface_idx = (interface_dim - 1 - y) * interface_dim + x;
			int mode_idx      = (mesh_dim      - 1 - j) * mesh_dim      + i;

			h_interface[interface_idx]   = h_raster[mode_idx];
			qx_interface[interface_idx]  = qx_raster[mode_idx];
			qy_interface[interface_idx]  = qy_raster[mode_idx];
			dem_interface[interface_idx] = dem[mode_idx];
		}
	}

	size_t bytes = interface_dim * interface_dim * sizeof(real);

	copy_cuda(d_nodal_vals.h,  h_interface,   bytes);
	copy_cuda(d_nodal_vals.qx, qx_interface,  bytes);
	copy_cuda(d_nodal_vals.qy, qy_interface,  bytes);
	copy_cuda(d_nodal_vals.z,  dem_interface, bytes);

	const int num_blocks = get_num_blocks(d_assem_sol.length, THREADS_PER_BLOCK);

	modal_projections<<<num_blocks, THREADS_PER_BLOCK>>>
	(
		d_nodal_vals,
		d_assem_sol,
		solver_params,
		mesh_dim,
		interface_dim
	);

	delete[] h_raster;
	delete[] qx_raster;
	delete[] qy_raster;
	delete[] dem;
	
	delete[] h_interface;
	delete[] qx_interface;
	delete[] qy_interface;
	delete[] dem_interface;
}