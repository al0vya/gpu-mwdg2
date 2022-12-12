#include "../input/read_raster_file.h"

void read_raster_file
(
	const char* raster_filename,
	real*       raster_array,
	const int&  mesh_dim,
	const real& wall_height
)
{
	char buf[255];

	int nrows = 0;
	int ncols = 0;

	real dummy = C(0.0);

	FILE* fp = fopen(raster_filename, "r");

	if (NULL == fp)
	{
		fprintf(stdout, "No raster file found: %s, using default values.\n", raster_filename);
		return;
	}

	fscanf(fp, "%s %d", buf, &ncols);
	fscanf(fp, "%s %d", buf, &nrows);
	fscanf(fp, "%s %" NUM_FRMT, buf, &dummy);
	fscanf(fp, "%s %" NUM_FRMT, buf, &dummy);
	fscanf(fp, "%s %" NUM_FRMT, buf, &dummy);
	fscanf(fp, "%s %" NUM_FRMT, buf, &dummy);

	for (int j = 0; j < nrows; j++)
	{
		for (int i = 0; i < ncols; i++)
		{
			fscanf(fp, "%" NUM_FRMT, &dummy);

			bool nodata = ( fabs( dummy + C(9999.0) ) < C(1e-10) );

			raster_array[ (nrows - 1 - j) * mesh_dim + i ] = (nodata) ? wall_height : dummy;

			int a = 1;
		}
	}

	fclose(fp);
}