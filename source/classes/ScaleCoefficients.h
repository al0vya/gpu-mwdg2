#pragma once

#include "cuda_utils.cuh"

#include "real.h"
#include "SolverTypes.h"

typedef struct ScaleCoefficients
{
	real* eta0;
	real* qx0;
	real* qy0;
	real* z0;

	real* eta1x;
	real* qx1x;
	real* qy1x;
	real* z1x;

	real* eta1y;
	real* qx1y;
	real* qy1y;
	real* z1y;

	bool is_copy = false;

	ScaleCoefficients
	(
		const int& num_all_elems, 
		const int& solver_type
	)
	{
		size_t bytes = sizeof(real) * num_all_elems;
		
		eta0 = (real*)malloc_device(bytes);
		qx0  = (real*)malloc_device(bytes);
		qy0  = (real*)malloc_device(bytes);
		z0   = (real*)malloc_device(bytes);

		eta1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qx1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qy1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		z1x   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;

		eta1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qx1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		qy1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		z1y   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	}

	ScaleCoefficients(const ScaleCoefficients& original) { *this = original; is_copy = true; }

	~ScaleCoefficients()
	{
		if (!is_copy)
		{
			free_device(eta0);
			free_device(qx0);
			free_device(qy0);
			free_device(z0);

			free_device(eta1x);
			free_device(qx1x);
			free_device(qy1x);
			free_device(z1x);

			free_device(eta1y);
			free_device(qx1y);
			free_device(qy1y);
			free_device(z1y);
		}
	}

} ScaleCoefficients;