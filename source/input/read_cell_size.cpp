#include "read_cell_size.h"

real read_cell_size(const char* input_filename)
{
	char dem_filename_buf[128] = {'\0'};
	read_keyword_str(input_filename, "DEMfile", dem_filename_buf);

	int  dummy          = 0;
	char dummy_buf[128] = {'\0'};
	real cell_size      = C(0.0);

	FILE* fp = fopen(dem_filename_buf, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening DEM file for cell size, file: %s, line: %d.\n", __FILE__, __LINE__);
		exit(-1);
	}

	fscanf(fp, "%s %d", dummy_buf, &dummy);
	fscanf(fp, "%s %d", dummy_buf, &dummy);
	fscanf(fp, "%s %" NUM_FRMT, dummy_buf, &cell_size);
	fscanf(fp, "%s %" NUM_FRMT, dummy_buf, &cell_size);
	fscanf(fp, "%s %" NUM_FRMT, dummy_buf, &cell_size);

	fclose(fp);

	return cell_size;
}