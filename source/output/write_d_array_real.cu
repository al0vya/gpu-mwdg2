#include "write_d_array_real.cuh"

__host__
void write_d_array_real
(
	const char* dirroot,
	const char* filename,
	real*       d_array,
	const int&  array_length
)
{
	// allocating host array to copy_cuda device array to 
	real* h_array = new real[array_length];

	size_t bytes = array_length * sizeof(real);

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
		fprintf(stderr, "Error opening %s for writing real array to file", filename);
		exit(-1);
	}

	for (int i = 0; i < array_length; i++)
	{
		fprintf( fp, "%" NUM_FRMT " ", h_array[i] );
	}

	fclose(fp);

	delete[] h_array;
}