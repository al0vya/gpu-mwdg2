#pragma once

#include "../utilities/cuda_utils.cuh"

#include "../types/real.h"
#include "../types/HierarchyIndex.h"
#include "../types/SolverTypes.h"

typedef struct AssembledSolution
{
	real* h0;
	real* qx0;
	real* qy0;
	real* z0;

	real* h1x;
	real* qx1x;
	real* qy1x;
	real* z1x;
	
	real* h1y;
	real* qx1y;
	real* qy1y;
	real* z1y;
	
	HierarchyIndex* act_idcs;

	int* levels;
	int  length;
	bool is_copy_cuda = false;

	AssembledSolution(const int& num_finest_elems, const int& solver_type)
	{
		size_t bytes_real = num_finest_elems * sizeof(real);
		size_t bytes_int  = num_finest_elems * sizeof(HierarchyIndex);

		h0  = (real*)malloc_device(bytes_real);
		qx0 = (real*)malloc_device(bytes_real);
		qy0 = (real*)malloc_device(bytes_real);
		z0  = (real*)malloc_device(bytes_real);

		h1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		qx1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		qy1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		z1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;

		h1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		qx1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		qy1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;
		z1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes_real) : nullptr;

		act_idcs = (HierarchyIndex*)malloc_device(bytes_int);
		levels   = (int*)malloc_device(bytes_int);
		length   = num_finest_elems;
	}

	AssembledSolution(const AssembledSolution& original) { *this = original; is_copy_cuda = true; }

	~AssembledSolution()
	{
		if (!is_copy_cuda)
		{
			CHECK_CUDA_ERROR( free_device(h0) );
			CHECK_CUDA_ERROR( free_device(qx0) );
			CHECK_CUDA_ERROR( free_device(qy0) );
			CHECK_CUDA_ERROR( free_device(z0) );

			CHECK_CUDA_ERROR( free_device(h1x) );
			CHECK_CUDA_ERROR( free_device(qx1x) );
			CHECK_CUDA_ERROR( free_device(qy1x) );
			CHECK_CUDA_ERROR( free_device(z1x) );

			CHECK_CUDA_ERROR( free_device(h1y) );
			CHECK_CUDA_ERROR( free_device(qx1y) );
			CHECK_CUDA_ERROR( free_device(qy1y) );
			CHECK_CUDA_ERROR( free_device(z1y) );

			CHECK_CUDA_ERROR( free_device(act_idcs) );
			CHECK_CUDA_ERROR( free_device(levels) );
		}
	}

} AssembledSolution;