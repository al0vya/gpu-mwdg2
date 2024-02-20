#include "write_hierarchy_array_bool.cuh"

__host__
void write_hierarchy_array_bool
(
	const char* dirroot,
	const char* filename,
	bool*       d_hierarchy,
	const int&  levels
)
{
	const int num_all_elems = get_lvl_idx(levels + 1);
	
	// allocating host array to copy_cuda device array to 
	bool* h_hierarchy = new bool[num_all_elems] ;

	size_t bytes = num_all_elems * sizeof(bool);

	copy_cuda
	(
		h_hierarchy,
		d_hierarchy,
		bytes
	);

	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%c%s%s", dirroot, '/', filename, ".txt");

	FILE* fp = fopen(fullpath, "w");

	if (NULL == fp)
	{
		fprintf(stderr, "Error opening %s for writing hierarchy to file", filename);
		exit(-1);
	}

	for (int level = 0; level <= levels; level++)
	{
		const int start = get_lvl_idx(level);
		const int end   = get_lvl_idx(level+1);

		for (int i = start; i < end; i++)
		{
			fprintf(fp, "%d ", h_hierarchy[i]);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);

	delete[] h_hierarchy;
}