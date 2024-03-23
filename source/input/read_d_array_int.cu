#include "read_d_array_int.cuh"

int* read_d_array_int
(
	const int&  num_items,
	const char* dirroot,
	const char* filename
)
{
	// setting up host and device hierarchy arrays
	const size_t bytes   = sizeof(int) * num_items;
	      int*   h_array = new int[num_items];
	      int*   d_array = (int*)malloc_device(bytes);
	
	// reading into host array from file
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s for reading int array.\n", fullpath);
		fclose(fp);
		exit(-1);
	}

	int dummy = 0;

	for (int i = 0; i < num_items; i++)
	{
		fscanf(fp, "%d", &dummy);

		h_array[i] = dummy;
	}

	// copying host hierarchy to device hierarchy
	copy_cuda(d_array, h_array, bytes);

	fclose(fp);

	delete[] h_array;

	return d_array;
}