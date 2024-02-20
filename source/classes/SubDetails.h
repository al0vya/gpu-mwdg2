#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

#include "../classes/SolverParams.h"

typedef struct SubDetails
{
	real* alpha;
	real* beta;
	real* gamma;

	int   levels = 0;
	bool  is_copy_cuda = false;

	SubDetails() = default;

	SubDetails
	(
		const SolverParams& solver_params
	)
		: levels(solver_params.L)
	{
		const int num_details = get_lvl_idx(this->levels);
		
		size_t bytes = sizeof(real) * num_details;

		alpha = (num_details > 0) ? (real*)malloc_device(bytes) : nullptr;
		beta  = (num_details > 0) ? (real*)malloc_device(bytes) : nullptr;
		gamma = (num_details > 0) ? (real*)malloc_device(bytes) : nullptr;
	}

	SubDetails(const SubDetails& original) { *this = original; is_copy_cuda = true; }

	~SubDetails()
	{
		if (!is_copy_cuda)
		{
			CHECK_CUDA_ERROR( free_device(alpha) );
			CHECK_CUDA_ERROR( free_device(beta) );
			CHECK_CUDA_ERROR( free_device(gamma) );
		}
	}

} SubDetails;