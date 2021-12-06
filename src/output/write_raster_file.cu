#include "write_raster_file.cuh"

__host__
void write_raster_file
(
	const char*                 respath,
	const char*                 file_extension,
	real*                       d_raster_array,
	const SimulationParams& sim_params,
	const SaveInterval          massint,
	const real&                 dx_finest,
	const int&                  mesh_dim
)
{
	char fullpath[255];

	sprintf(fullpath, "%s%s%d%c%s", respath, "results-", massint.count - 1, '.', file_extension);

	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file to write raster: %s", file_extension);
		exit(-1);
	}
	
	size_t bytes = mesh_dim * mesh_dim * sizeof(real);
	
	real xllcorner = sim_params.xmin;
	real yllcorner = sim_params.ymin;

	int ncols = sim_params.xsz;
	int nrows = sim_params.ysz;

	int NODATA_value = -9999;
	
	fprintf(fp, "ncols        %d\n", ncols);
	fprintf(fp, "nrows        %d\n", nrows);
	fprintf(fp, "xllcorner    %" NUM_FRMT "\n", xllcorner);
	fprintf(fp, "yllcorner    %" NUM_FRMT "\n", yllcorner);
	fprintf(fp, "cellsize     %" NUM_FRMT "\n", dx_finest);
	fprintf(fp, "NODATA_value %d\n", NODATA_value);

	real* raster_buf = new real[mesh_dim * mesh_dim];

	copy
	(
		raster_buf,
		d_raster_array,
		bytes
	);

	for (int j = 0; j < nrows; j++)
	{
		for (int i = 0; i < ncols; i++)
		{
			int idx = (nrows - 1 - j) * mesh_dim + i;

			fprintf(fp, "%" NUM_FIG NUM_FRMT " ", raster_buf[idx]);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);

	delete[] raster_buf;
}