#include "write_hierarchy_to_file.cuh"

__host__
void write_hierarchy_to_file
(
	const char* filename,
	real*       d_hierarchy,
	const int&  levels
)
{
	const int num_all_elems = get_lvl_idx(levels + 1);
	
	// allocating host array to copy_cuda device array to 
	real* h_hierarchy = new real[num_all_elems];

	size_t bytes = num_all_elems * sizeof(real);

	copy_cuda
	(
		h_hierarchy,
		d_hierarchy,
		bytes
	);

	char fullpath[255] = {'\0'};

	sprintf(fullpath, "%s%s", filename, ".txt");

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
			fprintf(fp, "%" NUM_FRMT " ", h_hierarchy[i]);
		}

		fprintf(fp, "\n");
	}

	fclose(fp);

	delete[] h_hierarchy;
}