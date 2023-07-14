#include "write_reals_to_file.cuh"

__host__
void write_reals_to_file
(
	const char* filename,
	const char* dirroot,
	real*       d_results,
	const int&  array_length
)
{
	// allocating host array to copy_cuda device array to 
	real* h_results = new real[array_length];

	size_t bytes = array_length * sizeof(real);

	copy_cuda
	(
		h_results,
		d_results,
		bytes
	);

	FILE* fp;

	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%s%s", dirroot, filename, ".csv");

	fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file : %s", filename);
		exit(-1);
	}

	fprintf(fp, "results\n");

	for (int i = 0; i < array_length; i++)
	{
		fprintf(fp, "%" NUM_FRMT "\n", h_results[i]);
	}

	fclose(fp);

	delete[] h_results;
}