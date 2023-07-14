#include "write_bool_to_file.cuh"

__host__
void write_bool_to_file
(
	const char* filename,
	const char* dirroot,
	bool*       d_results,
	const int   array_length
)
{
	// allocating host array to copy_cuda device array to 
	bool* h_results = new bool[array_length];

	size_t bytes = array_length * sizeof(bool);

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
		fprintf(stderr, "Error opening file: %s", filename);
		exit(-1);
	}

	fprintf(fp, "results\n");

	for (int i = 0; i < array_length; i++)
	{
		fprintf(fp, "%d\n", h_results[i]);
	}

	fclose(fp);

	delete[] h_results;
}