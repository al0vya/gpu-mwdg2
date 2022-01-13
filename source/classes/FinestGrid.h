#pragma once

#include "cuda_utils.cuh"

#include "real.h"

typedef struct FinestGrid
{
    real* h;
    real* qx;
    real* qy;
    real* z;
	
	bool is_copy = false;
	
	FinestGrid(const int& num_finest_elems)
	{
	    size_t bytes = sizeof(real) * num_finest_elems;
		
		h  = (real*)malloc_pinned(bytes);
		qx = (real*)malloc_pinned(bytes);
		qy = (real*)malloc_pinned(bytes);
		z  = (real*)malloc_pinned(bytes);
	}
	
	FinestGrid(const FinestGrid& original) { *this = original; is_copy = true; }
	
	~FinestGrid()
	{
	    if (!is_copy)
		{
		    CHECK_CUDA_ERROR( free_pinned(h)  );
		    CHECK_CUDA_ERROR( free_pinned(qx) );
		    CHECK_CUDA_ERROR( free_pinned(qy) );
		    CHECK_CUDA_ERROR( free_pinned(z)  );
		}
	}

} FinestGrid;