#include "read_hierarchy_array_real.cuh"

real* read_hierarchy_array_real
(
	const int&  levels,
	const char* dirroot,
	const char* filename
)
{
	// setting up host and device hierarchy arrays
	const int    num_all_elems = get_lvl_idx(levels + 1);
	const size_t bytes         = sizeof(real) * num_all_elems;
	      real*  h_hierarchy   = new real[num_all_elems];
	      real*  d_hierarchy   = (real*)malloc_device(bytes);
	
	// reading into host array from file
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening file %s for reading hierarchy array.\n", fullpath);
		fclose(fp);
		exit(-1);
	}

	real dummy = C(0.0);

	for (int i = 0; i < num_all_elems; i++)
	{
		fscanf(fp, "%" NUM_FRMT, &dummy);
		
		h_hierarchy[i] = dummy;
	}

	// copying host hierarchy to device hierarchy
	copy_cuda(d_hierarchy, h_hierarchy, bytes);

	fclose(fp);

	delete[] h_hierarchy;

	return d_hierarchy;
}