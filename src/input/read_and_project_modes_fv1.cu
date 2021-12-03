#include "read_and_project_modes_fv1.cuh"

__host__
void read_and_project_modes_fv1
(
	const char*              input_filename,
	const AssembledSolution& d_assem_sol,
	const int&               mesh_dim,
	const real&              wall_height
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

	size_t bytes = d_assem_sol.length * sizeof(real);

	for (int i = 0; i < d_assem_sol.length; i++) dem[i] = wall_height;

	read_raster_file(h_raster_filename,  h_raster,  mesh_dim, wall_height);
	read_raster_file(qx_raster_filename, qx_raster, mesh_dim, wall_height);
	read_raster_file(qy_raster_filename, qy_raster, mesh_dim, wall_height);
	read_raster_file(dem_filename,       dem,       mesh_dim, wall_height);

	copy(d_assem_sol.h0,  h_raster,  bytes);
	copy(d_assem_sol.qx0, qx_raster, bytes);
	copy(d_assem_sol.qy0, qy_raster, bytes);
	copy(d_assem_sol.z0,  dem,       bytes);

	delete[] h_raster;
	delete[] qx_raster;
	delete[] qy_raster;
	delete[] dem;
}