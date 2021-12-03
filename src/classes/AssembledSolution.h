#pragma once

#include "cuda_utils.cuh"

#include "real.h"
#include "HierarchyIndex.h"
#include "SolverTypes.h"

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
	bool is_copy = false;

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

	AssembledSolution(const AssembledSolution& original) { *this = original; is_copy = true; }

	~AssembledSolution()
	{
		if (!is_copy)
		{
			free_device(h0);
			free_device(qx0);
			free_device(qy0);
			free_device(z0);

			free_device(h1x);
			free_device(qx1x);
			free_device(qy1x);
			free_device(z1x);
			
			free_device(h1y);
			free_device(qx1y);
			free_device(qy1y);
			free_device(z1y);
			
			free_device(act_idcs);
			free_device(levels);
		}
	}

} AssembledSolution;