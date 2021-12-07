#include "read_cell_size.h"

real read_cell_size(const char* input_filename)
{
	int dummy = 0;

	real cell_size = C(0.0);

	char str[255];
	char buf[64];
	char rasterroot[64];
	char dem_filename[64];
	
	FILE* fp = fopen(input_filename, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening input file for reading raster root name for cell size.\n");
		exit(-1);
	}

	while ( strncmp(buf, "rasterroot", 10) )
	{
		if ( NULL == fgets(str, sizeof(str), fp) )
		{
			fprintf(stderr, "Error reading input file for reading raster root name for cell size.\n");
			fclose(fp);
			exit(-1);
		}

		sscanf(str, "%s %s", buf, rasterroot);
	}

	fclose(fp);

	sprintf(dem_filename, "%s%s", rasterroot, ".dem");

	fp = fopen(dem_filename, "r");

	if ( NULL == fgets(str, sizeof(str), fp) )
	{
		fprintf(stderr, "Error opening DEM file for cell size.\n");
		fclose(fp);
		exit(-1);
	}

	fscanf(fp, "%s %d", buf, &dummy);
	fscanf(fp, "%s %d", buf, &dummy);
	fscanf(fp, "%s %" NUM_FRMT, buf, &cell_size);
	fscanf(fp, "%s %" NUM_FRMT, buf, &cell_size);
	fscanf(fp, "%s %" NUM_FRMT, buf, &cell_size);

	fclose(fp);

	return cell_size;
}