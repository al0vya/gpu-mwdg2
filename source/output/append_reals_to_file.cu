#include "append_reals_to_file.cuh"

__host__
void append_reals_to_file
(
	const char* filename,
	const char* respath,
	real*       d_results,
	const int&  array_length,
	const bool  first_t_step
)
{
	// allocating host array to copy device array to 
	real* h_results = new real[array_length];

	size_t bytes = array_length * sizeof(real);

	copy
	(
		h_results,
		d_results,
		bytes
	);

	FILE* fp;

	char fullpath[255];

	sprintf(fullpath, "%s%s%s", respath, filename, ".csv");

	fp = (first_t_step) ? fopen(fullpath, "w") : fopen(fullpath, "a");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file : %s", filename);
		exit(-1);
	}

	for (int i = 0; i < array_length; i++)
	{
		fprintf(fp, "%" NUM_FIG NUM_FRMT ",", h_results[i]);
	}
	
	fprintf(fp, "\n");

	fclose(fp);

	delete[] h_results;
}