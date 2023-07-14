#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"

typedef struct NodalValues
{
	real* h;
	real* qx;
	real* qy;
	real* z;
	bool  is_copy_cuda = false;

	NodalValues(const int& interface_dim)
	{
		size_t bytes = sizeof(real) * interface_dim * interface_dim;

		h  = (real*)malloc_device(bytes);
		qx = (real*)malloc_device(bytes);
		qy = (real*)malloc_device(bytes);
		z  = (real*)malloc_device(bytes);
	}

	NodalValues(const NodalValues& original) { *this = original; is_copy_cuda = true; }

	~NodalValues()
	{
		if (!is_copy_cuda)
		{
			CHECK_CUDA_ERROR( free_device(h)  );
			CHECK_CUDA_ERROR( free_device(qx) );
			CHECK_CUDA_ERROR( free_device(qy) );
			CHECK_CUDA_ERROR( free_device(z)  );
		}
	}

} NodalValues;