#include "read_and_project_modes_dg2.cuh"

__host__
void read_and_project_modes_dg2
(
	const char*                 input_filename,
	const AssembledSolution&    d_assem_sol,
	const NodalValues&          d_nodal_vals,
	const SimulationParameters& sim_params,
	const SolverParameters&     solver_params,
	const int&                  mesh_dim
)
{
	FILE* fp = fopen(input_filename, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening input file for raster root name.\n");
		exit(-1);
	}

	char str[255]       = {'\0'};
	char buf[64]        = {'\0'};
	char rasterroot[64] = {'\0'};

	while ( strncmp(buf, "rasterroot", 10) )
	{
		if ( NULL == fgets(str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for raster root name.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %s", buf, rasterroot);
	}

	char h_raster_filename[64]  = {'\0'};
	char qx_raster_filename[64] = {'\0'};
	char qy_raster_filename[64] = {'\0'};
	char dem_filename[64]       = {'\0'};

	sprintf(h_raster_filename,  "%s%s", rasterroot, ".start");
	sprintf(qx_raster_filename, "%s%s", rasterroot, ".start.Qx");
	sprintf(qy_raster_filename, "%s%s", rasterroot, ".start.Qy");
	sprintf(dem_filename,       "%s%s", rasterroot, ".dem");

	real* h_raster  = new real[d_assem_sol.length]();
	real* qx_raster = new real[d_assem_sol.length]();
	real* qy_raster = new real[d_assem_sol.length]();
	real* dem       = new real[d_assem_sol.length]();

	for (int i = 0; i < d_assem_sol.length; i++) dem[i] = solver_params.wall_height;

	read_raster_file(h_raster_filename,  h_raster,  mesh_dim, solver_params.wall_height);
	read_raster_file(qx_raster_filename, qx_raster, mesh_dim, solver_params.wall_height);
	read_raster_file(qy_raster_filename, qy_raster, mesh_dim, solver_params.wall_height);
	read_raster_file(dem_filename,       dem,       mesh_dim, solver_params.wall_height);
	
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

	copy(d_nodal_vals.h,  h_interface,   bytes);
	copy(d_nodal_vals.qx, qx_interface,  bytes);
	copy(d_nodal_vals.qy, qy_interface,  bytes);
	copy(d_nodal_vals.z,  dem_interface, bytes);

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