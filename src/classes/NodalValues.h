#pragma once

#include "cuda_utils.cuh"

#include "real.h"

typedef struct NodalValues
{
	real* h;
	real* qx;
	real* qy;
	real* z;
	bool  is_copy = false;

	NodalValues(const int& interface_dim)
	{
		size_t bytes = sizeof(real) * interface_dim * interface_dim;

		h  = (real*)malloc_device(bytes);
		qx = (real*)malloc_device(bytes);
		qy = (real*)malloc_device(bytes);
		z  = (real*)malloc_device(bytes);
	}

	NodalValues(const NodalValues& original) { *this = original; is_copy = true; }

	~NodalValues()
	{
		if (!is_copy)
		{
			free_device(h);
			free_device(qx);
			free_device(qy);
			free_device(z);
		}
	}

} NodalValues;