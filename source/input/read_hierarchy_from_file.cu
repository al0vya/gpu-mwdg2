#include "read_hierarchy_from_file.cuh"

void read_hierarchy_from_file
(
	      real* d_hierarchy,
	const int&  levels,
	const char* dirroot,
	const char* filename
)
{
	// setting up host hierarchy array
	const int num_all_elems = get_lvl_idx(levels + 1);
	const size_t bytes      = sizeof(real) * num_all_elems;
	real* h_hierarchy       = new real[num_all_elems];

	// reading into host array from file
	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "r");

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
}