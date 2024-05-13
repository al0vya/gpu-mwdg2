#pragma once

#include <string>

#include "../utilities/cuda_utils.cuh"
#include "../utilities/compare_d_array_with_file_int.cuh"
#include "../utilities/compute_max_error.cuh"
#include "../utilities/zero_array_kernel_int.cuh"
#include "../utilities/zero_array_kernel_real.cuh"
#include "../utilities/compute_max_error.cuh"
#include "../types/HierarchyIndex.h"
#include "../classes/SolverParams.h"
#include "../input/read_d_array_int.cuh"
#include "../input/read_d_array_real.cuh"
#include "../output/write_d_array_int.cuh"
#include "../output/write_d_array_real.cuh"

class AssembledSolution
{
public:
	AssembledSolution
	(
		const SolverParams& solver_params
	);

	AssembledSolution
	(
		const SolverParams& solver_params,
		const std::string&  name
	);
	
	AssembledSolution
	(
		const SolverParams& solver_params,
		const char*         dirroot,
		const char*         prefix
	);

	AssembledSolution
	(
		const SolverParams& solver_params,
		const std::string&  name,
		const char*         dirroot,
		const char*         prefix
	);

	AssembledSolution
	(
		const AssembledSolution& original
	);

	~AssembledSolution();

	void write_to_file
	(
		const char* dirroot,
		const char* prefix
	);
    
	real verify_real
	(
		const char* dirroot,
		const char* prefix
	);

	int verify_int
	(
		const char* dirroot,
		const char* prefix
	);

	real* h0  = nullptr;
	real* qx0 = nullptr;
	real* qy0 = nullptr;
	real* z0  = nullptr;

	real* h1x  = nullptr;
	real* qx1x = nullptr;
	real* qy1x = nullptr;
	real* z1x  = nullptr;

	real* h1y  = nullptr;
	real* qx1y = nullptr;
	real* qy1y = nullptr;
	real* z1y  = nullptr; 

	HierarchyIndex* act_idcs = nullptr;

	int* levels = nullptr;
	bool* wet_cells = nullptr;
	
	int length = 0;
	int max_length = 0;
	int solver_type = 0;
	bool is_copy_cuda = false;
	std::string name = "";

private:
	void initialise();

	void initialise_from_file
	(
		const char* dirroot,
		const char* prefix
	);
};