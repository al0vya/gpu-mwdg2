#include "write_raster_file.cuh"

__host__
void write_raster_file
(
	const PlottingParams&   plot_params,
	const char*             file_extension,
	real*                   raster,
	const SimulationParams& sim_params,
	const SaveInterval      saveint,
	const real&             dx_finest,
	const int&              mesh_dim
)
{
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%c%d%c%s", plot_params.dirroot, '/', plot_params.resroot, '-', saveint.count - 1, '.', file_extension);

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

	for (int j = 0; j < nrows; j++)
	{
		for (int i = 0; i < ncols; i++)
		{
			int idx = (nrows - 1 - j) * mesh_dim + i;

			fprintf(fp, "%" NUM_FIG NUM_FRMT " ", raster[idx]);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);
}