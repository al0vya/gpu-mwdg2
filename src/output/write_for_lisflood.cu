#include "write_for_lisflood.cuh"

__host__ void write_for_lisflood
(
	const char*                 respath,
	const AssembledSolution&    d_assem_sol,
	const int&                  mesh_dim,
	const real&                 dx_finest,
	const SimulationParams& sim_params
)
{
	size_t bytes = mesh_dim * mesh_dim * sizeof(real);
	
	real xllcorner = sim_params.xmin;
	real yllcorner = sim_params.ymin;

	real cellsize = dx_finest;

	int NODATA_value = -9999;
	
	FILE* fp;

	char fullpath[255];

	// WRITING DEM //

	sprintf(fullpath, "%s%s", respath, "lisflood.dem");

	fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file: %s", "lisflood.dem");
		exit(-1);
	}

	fprintf(fp, "ncols        %d\n", mesh_dim);
	fprintf(fp, "nrows        %d\n", mesh_dim);
	fprintf(fp, "xllcorner    %" NUM_FRMT "\n", xllcorner);
	fprintf(fp, "yllcorner    %" NUM_FRMT "\n", yllcorner);
	fprintf(fp, "cellsize     %" NUM_FRMT "\n", cellsize);
	fprintf(fp, "NODATA_value %d\n", NODATA_value);

	real* DEM = new real[mesh_dim * mesh_dim];

	copy
	(
		DEM,
		d_assem_sol.z0,
		bytes
	);

	for (int j = 0; j < mesh_dim; j++)
	{
		for (int i = 0; i < mesh_dim; i++)
		{
			int idx = j * mesh_dim + i;

			fprintf(fp, "%.15" NUM_FRMT " ", DEM[idx]);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);

	delete[] DEM;

	// ----------- //

	// reset the string https://stackoverflow.com/questions/1559487/how-to-empty-a-char-array
	fullpath[0] = '\0';

	// WRITING WATER DEPTH "H" //
	
	sprintf(fullpath, "%s%s", respath, "lisflood.start");


	fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file : %s", "lisflood.start");
		exit(-1);
	}

	fprintf(fp, "ncols        %d\n", mesh_dim);
	fprintf(fp, "nrows        %d\n", mesh_dim);
	fprintf(fp, "xllcorner    %" NUM_FRMT "\n", xllcorner);
	fprintf(fp, "yllcorner    %" NUM_FRMT "\n", yllcorner);
	fprintf(fp, "cellsize     %" NUM_FRMT "\n", cellsize);
	fprintf(fp, "NODATA_value %d\n", NODATA_value);

	real* depth = new real[mesh_dim * mesh_dim];

	copy
	(
		depth,
		d_assem_sol.h0,
		bytes
	);

	for (int j = 0; j < mesh_dim; j++)
	{
		for (int i = 0; i < mesh_dim; i++)
		{
			int idx = j * mesh_dim + i;

			real h = depth[idx];

			fprintf(fp, "%.15" NUM_FRMT " ", ( h < C(0.0) ) ? C(0.0) : h);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);

	delete[] depth;

	// ----------- //

}