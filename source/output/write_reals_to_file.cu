#include "write_reals_to_file.cuh"

__host__
void write_reals_to_file
(
	const char* filename,
	const char* respath,
	real*       d_results,
	const int&  array_length
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

	fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file : %s", filename);
		exit(-1);
	}

	fprintf(fp, "results\n");

	real sum = 0;

	for (int i = 0; i < array_length; i++)
	{
		fprintf(fp, "%" NUM_FIG NUM_FRMT "\n", h_results[i]);

		sum += h_results[i];
	}

	printf("%s: %" NUM_FIG NUM_FRMT "\n", filename, sum / array_length);

	fclose(fp);

	delete[] h_results;
}