#include "write_d_array_int.cuh"

__host__
void write_d_array_int
(
	const char* dirroot,
	const char* filename,
	int*        d_array,
	const int&  array_length
)
{
	// allocating host array to copy_cuda device array to 
	int* h_array = new int[array_length];

	size_t bytes = array_length * sizeof(int);

	copy_cuda
	(
		h_array,
		d_array,
		bytes
	);

	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening %s for writing int array to file", filename);
		exit(-1);
	}

	for (int i = 0; i < array_length; i++)
	{
		fprintf( fp, "%d ", h_array[i] );
	}

	fclose(fp);

	delete[] h_array;
}