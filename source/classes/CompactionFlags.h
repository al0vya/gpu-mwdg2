#pragma once

#include "cuda_utils.cuh"

typedef struct CompactionFlags
{
	bool* north_east;
	bool* south_west;
	bool  is_copy = false;

	CompactionFlags(const int& num_finest_elems)
	{
		size_t bytes = sizeof(bool) * num_finest_elems;

		north_east = (bool*)malloc_device(bytes);
		south_west = (bool*)malloc_device(bytes);
	}

	CompactionFlags(const CompactionFlags& original) { *this = original; is_copy = true; }

	~CompactionFlags()
	{
		if (!is_copy)
		{
			free_device(north_east);
			free_device(south_west);
		}
	}

} CompactionFlags;