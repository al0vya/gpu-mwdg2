#include "read_hierarchy_array_bool.cuh"

bool* read_hierarchy_array_bool
(
	const int&  levels,
	const char* dirroot,
	const char* filename
)
{
	// setting up host and device hierarchy arrays
	const int    num_all_elems = get_lvl_idx(levels + 1);
	const size_t bytes         = sizeof(bool) * num_all_elems;
	      bool*  h_hierarchy   = new bool[num_all_elems];
	      bool*  d_hierarchy   = (bool*)malloc_device(bytes);
	
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

	bool dummy = 0;

	for (int i = 0; i < num_all_elems; i++)
	{
		if (i < PADDING_MRA) // padded section of hierarchy array should be zero
		{
			h_hierarchy[i] = 0;
		}
		else
		{
			fscanf(fp, "%d", &dummy);

			h_hierarchy[i] = dummy;
		}
	}

	// copying host hierarchy to device hierarchy
	copy_cuda(d_hierarchy, h_hierarchy, bytes);

	fclose(fp);

	delete[] h_hierarchy;

	return d_hierarchy;
}