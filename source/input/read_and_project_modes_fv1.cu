#include "../input/read_and_project_modes_fv1.cuh"

__host__
void read_and_project_modes_fv1
(
	const char*              input_filename,
	const AssembledSolution& d_assem_sol,
	const int&               mesh_dim,
	const SolverParams&      solver_params
)
{
	char h_raster_filename_buf[128]  = {'\0'};
	char qx_raster_filename_buf[128] = {'\0'};
	char qy_raster_filename_buf[128] = {'\0'};
	char dem_filename_buf[128]       = {'\0'};

	read_keyword_str(input_filename, "DEMfile", dem_filename_buf);
	
	// buffer for DEM filename should never be null char because DEM is always needed for realistic test case
	if (dem_filename_buf[0] == '\0')
	{
		fprintf(stderr, "Error reading DEM filename, file: %s, line: %d.\n", __FILE__, __LINE__);
	}
	
	read_keyword_str(input_filename, "startfile", h_raster_filename_buf);

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

	size_t bytes = d_assem_sol.length * sizeof(real);

	copy_cuda(d_assem_sol.h0,  h_raster,  bytes);
	copy_cuda(d_assem_sol.qx0, qx_raster, bytes);
	copy_cuda(d_assem_sol.qy0, qy_raster, bytes);
	copy_cuda(d_assem_sol.z0,  dem,       bytes);

	delete[] h_raster;
	delete[] qx_raster;
	delete[] qy_raster;
	delete[] dem;
}