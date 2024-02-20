#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

#include "../classes/SolverParams.h"

#include "../input/read_hierarchy_array_real.cuh"
#include "../output/write_hierarchy_array_real.cuh"

typedef struct SubDetails
{
	real* alpha = nullptr;
	real* beta  = nullptr;
	real* gamma = nullptr;

	int   levels       = 0;
	bool  is_copy_cuda = false;

	SubDetails() = default;

	SubDetails
	(
		const SolverParams& solver_params
	)
		: levels(solver_params.L - 1)
	{
		const int num_details = get_lvl_idx(this->levels + 1);
		
		size_t bytes = sizeof(real) * num_details;

		// no allocation if defaultly constructed SolverParams is passed since
		// solver_params.L - 1 = 0 so this->levels will be 0
		alpha = (this->levels > 0) ? (real*)malloc_device(bytes) : nullptr;
		beta  = (this->levels > 0) ? (real*)malloc_device(bytes) : nullptr;
		gamma = (this->levels > 0) ? (real*)malloc_device(bytes) : nullptr;
	}

	SubDetails
	(
		const SolverParams& solver_params,
		const char*         dirroot,
		const char*         suffix
	)
		: levels(solver_params.L - 1)
	{
		const int num_details = get_lvl_idx(this->levels + 1);
		
		char filename_alpha[255] = {'\0'};
		char filename_beta [255] = {'\0'};
		char filename_gamma[255] = {'\0'};

		sprintf(filename_alpha, "%s%c%s", "output-details-alpha", '-', suffix);
		sprintf(filename_beta , "%s%c%s", "output-details-beta",  '-', suffix);
		sprintf(filename_gamma, "%s%c%s", "output-details-gamma", '-', suffix);

		// no allocation if defaultly constructed SolverParams is passed since
		// solver_params.L - 1 = 0 so this->levels will be 0
		alpha = (this->levels > 0) ? read_hierarchy_array_real(this->levels, dirroot, filename_alpha) : nullptr;
		beta  = (this->levels > 0) ? read_hierarchy_array_real(this->levels, dirroot, filename_beta)  : nullptr;
		gamma = (this->levels > 0) ? read_hierarchy_array_real(this->levels, dirroot, filename_gamma) : nullptr;
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

	void write_to_file
	(
		const char* dirroot,
		const char* prefix,
		const char* suffix
	)
	{
		char filename_alpha[255] = {'\0'};
		char filename_beta [255] = {'\0'};
		char filename_gamma[255] = {'\0'};

		sprintf(filename_alpha, "%s%c%s%c%s", prefix, '-', "details-alpha", '-', suffix);
		sprintf(filename_beta , "%s%c%s%c%s", prefix, '-', "details-beta",  '-', suffix);
		sprintf(filename_gamma, "%s%c%s%c%s", prefix, '-', "details-gamma", '-', suffix);

		write_hierarchy_array_real(dirroot, filename_alpha, this->alpha, this->levels);
		write_hierarchy_array_real(dirroot, filename_beta,  this->beta,  this->levels);
		write_hierarchy_array_real(dirroot, filename_gamma, this->gamma, this->levels);
	}

} SubDetails;