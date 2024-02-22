#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/compute_error.cuh"

#include "../classes/SolverParams.h"

#include "../input/read_hierarchy_array_real.cuh"
#include "../output/write_hierarchy_array_real.cuh"

class SubDetails
{
public:
	SubDetails();

	SubDetails
	(
		const int& levels
	);

	SubDetails
	(
		const int&  levels,
		const char* dirroot,
		const char* prefix,
		const char* suffix
	);

	SubDetails
	(
		const SubDetails& original
	);

	~SubDetails();

	void write_to_file
	(
		const char* dirroot,
		const char* prefix,
		const char* suffix
	);

	real verify
	(
		const char* dirroot,
		const char* prefix,
		const char* suffix
	);

	real* alpha = nullptr;
	real* beta  = nullptr;
	real* gamma = nullptr;

	int   levels       = 0;
	bool  is_copy_cuda = false;
};