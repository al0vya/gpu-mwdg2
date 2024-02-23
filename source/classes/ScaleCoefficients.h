#pragma once

#include <algorithm>

#include "../classes/SolverParams.h"
#include "../utilities/get_lvl_idx.cuh"
#include "../utilities/compute_error.cuh"
#include "../output/write_hierarchy_array_real.cuh"
#include "../input/read_hierarchy_array_real.cuh"

class ScaleCoefficients
{
public:
	ScaleCoefficients
	(
		const SolverParams& solver_params
	);
    
	ScaleCoefficients
	(
		const SolverParams& solver_params,
		const char*         dirroot,
		const char*         prefix
	);
    
	ScaleCoefficients
	(
		const ScaleCoefficients& original
	);
	
	~ScaleCoefficients();

    void write_to_file
	(
		const char* dirroot,
		const char* prefix
	);
    
	real verify
	(
		const char* dirroot,
		const char* prefix
	);
	
	real* eta0 = nullptr;
	real* qx0  = nullptr;
	real* qy0  = nullptr;
	real* z0   = nullptr;

	real* eta1x = nullptr;
	real* qx1x  = nullptr;
	real* qy1x  = nullptr;
	real* z1x   = nullptr;

	real* eta1y = nullptr;
	real* qx1y  = nullptr;
	real* qy1y  = nullptr;
	real* z1y   = nullptr;

	bool is_copy_cuda = false;

	int levels = 0;
	int solver_type = 0;
};