#include "read_hierarchy_array_int.cuh"

int* read_hierarchy_array_int
(
	const int&  levels,
	const char* dirroot,
	const char* filename
)
{
	// setting up host and device hierarchy arrays
	const int    num_all_elems = get_lvl_idx(levels + 1);
	const size_t bytes         = sizeof(int) * num_all_elems;
	      int*  h_hierarchy    = new int[num_all_elems];
	      int*  d_hierarchy    = (int*)malloc_device(bytes);
	
	// reading into host array from file
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

	int dummy = 0;

	for (int i = 0; i < num_all_elems; i++)
	{
		fscanf(fp, "%d", &dummy);
		
		h_hierarchy[i] = dummy;
	}

	// copying host hierarchy to device hierarchy
	copy_cuda(d_hierarchy, h_hierarchy, bytes);

	fclose(fp);

	delete[] h_hierarchy;

	return d_hierarchy;
}