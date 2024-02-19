#pragma once

#include "../utilities/cuda_utils.cuh"
#include "../utilities/get_lvl_idx.cuh"

#include "../types/real.h"
#include "../types/SolverTypes.h"


#include "../output/write_hierarchy_to_file.cuh"
#include "../input/read_hierarchy_from_file.cuh"

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

	bool is_copy_cuda = false;

	int levels = 0;
	int solver_type = 0;

	ScaleCoefficients
	(
		const int& levels,
		const int& solver_type
	)
		: levels(levels), solver_type(solver_type)
	{
		const int num_all_elems = get_lvl_idx(this->levels + 1);
		
		const size_t bytes = sizeof(real) * num_all_elems;
		
		this->eta0 = (real*)malloc_device(bytes);
		this->qx0  = (real*)malloc_device(bytes);
		this->qy0  = (real*)malloc_device(bytes);
		this->z0   = (real*)malloc_device(bytes);

		this->eta1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qx1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qy1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->z1x   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;

		this->eta1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qx1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qy1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->z1y   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
	}

	ScaleCoefficients
	(
		const int&  levels,
		const int&  solver_type,
		const char* dirroot
	)
		: levels(levels), solver_type(solver_type)
	{
		const int num_all_elems = get_lvl_idx(levels + 1);
		
		const size_t bytes = sizeof(real) * num_all_elems;

		this->eta0 = (real*)malloc_device(bytes);
		this->qx0  = (real*)malloc_device(bytes);
		this->qy0  = (real*)malloc_device(bytes);
		this->z0   = (real*)malloc_device(bytes);
		
		this->eta1x = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qx1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qy1x  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->z1x   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		
		this->eta1y = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qx1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->qy1y  = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		this->z1y   = (solver_type == MWDG2) ? (real*)malloc_device(bytes) : nullptr;
		
		if (this->solver_type == HWFV1)
		{
			read_hierarchy_from_file(this->eta0, this->levels, dirroot, "input-scale-coeffs-eta0-hw");
			read_hierarchy_from_file(this->qx0,  this->levels, dirroot, "input-scale-coeffs-qx0-hw");
			read_hierarchy_from_file(this->qy0,  this->levels, dirroot, "input-scale-coeffs-qy0-hw");
			read_hierarchy_from_file(this->z0,   this->levels, dirroot, "input-scale-coeffs-z0-hw");
		}
		else if (this->solver_type == MWDG2) // reading from file for GPU-MWDG2
		{
			read_hierarchy_from_file(this->eta0,  this->levels, dirroot, "input-scale-coeffs-eta0-mw");
			read_hierarchy_from_file(this->qx0,   this->levels, dirroot, "input-scale-coeffs-qx0-mw");
			read_hierarchy_from_file(this->qy0,   this->levels, dirroot, "input-scale-coeffs-qy0-mw");
			read_hierarchy_from_file(this->z0,    this->levels, dirroot, "input-scale-coeffs-z0-mw");
			read_hierarchy_from_file(this->eta1x, this->levels, dirroot, "input-scale-coeffs-eta1x-mw");
			read_hierarchy_from_file(this->qx1x,  this->levels, dirroot, "input-scale-coeffs-qx1x-mw");
			read_hierarchy_from_file(this->qy1x,  this->levels, dirroot, "input-scale-coeffs-qy1x-mw");
			read_hierarchy_from_file(this->z1x,   this->levels, dirroot, "input-scale-coeffs-z1x-mw");
			read_hierarchy_from_file(this->eta1y, this->levels, dirroot, "input-scale-coeffs-eta1y-mw");
			read_hierarchy_from_file(this->qx1y,  this->levels, dirroot, "input-scale-coeffs-qx1y-mw");
			read_hierarchy_from_file(this->qy1y,  this->levels, dirroot, "input-scale-coeffs-qy1y-mw");
			read_hierarchy_from_file(this->z1y,   this->levels, dirroot, "input-scale-coeffs-z1y-mw");
		}
	}

	ScaleCoefficients(const ScaleCoefficients& original) { *this = original; is_copy_cuda = true; }

	~ScaleCoefficients()
	{
		if (!is_copy_cuda)
		{
			CHECK_CUDA_ERROR( free_device(eta0) );
			CHECK_CUDA_ERROR( free_device(qx0) );
			CHECK_CUDA_ERROR( free_device(qy0) );
			CHECK_CUDA_ERROR( free_device(z0) );

			CHECK_CUDA_ERROR( free_device(eta1x) );
			CHECK_CUDA_ERROR( free_device(qx1x) );
			CHECK_CUDA_ERROR( free_device(qy1x) );
			CHECK_CUDA_ERROR( free_device(z1x) );

			CHECK_CUDA_ERROR( free_device(eta1y) );
			CHECK_CUDA_ERROR( free_device(qx1y) );
			CHECK_CUDA_ERROR( free_device(qy1y) );
			CHECK_CUDA_ERROR( free_device(z1y) );
		}
	}

	void write_to_file
	(
		const char* dirroot,
		const char* prefix
	)
	{
		if (this->solver_type == HWFV1)
		{
			char filename_eta0[255] = {'\0'};
			char filename_qx0[255]  = {'\0'};
			char filename_qy0[255]  = {'\0'};
			char filename_z0[255]   = {'\0'};

			sprintf(filename_eta0, "%s%c%s", prefix, '-', "scale-coeffs-eta0-hw");
			sprintf(filename_qx0,  "%s%c%s", prefix, '-', "scale-coeffs-qx0-hw");
			sprintf(filename_qy0,  "%s%c%s", prefix, '-', "scale-coeffs-qy0-hw");
			sprintf(filename_z0,   "%s%c%s", prefix, '-', "scale-coeffs-z0-hw");
			
			write_hierarchy_to_file(dirroot, filename_eta0, this->eta0, this->levels);
			write_hierarchy_to_file(dirroot, filename_qx0,  this->qx0,  this->levels);
			write_hierarchy_to_file(dirroot, filename_qy0,  this->qy0,  this->levels);
			write_hierarchy_to_file(dirroot, filename_z0,   this->z0,   this->levels);
		}
		if (this->solver_type == MWDG2)
		{
			char filename_eta0[255]  = {'\0'};
			char filename_qx0[255]   = {'\0'};
			char filename_qy0[255]   = {'\0'};
			char filename_z0[255]    = {'\0'};
			char filename_eta1x[255] = {'\0'};
			char filename_qx1x[255]  = {'\0'};
			char filename_qy1x[255]  = {'\0'};
			char filename_z1x[255]   = {'\0'};
			char filename_eta1y[255] = {'\0'};
			char filename_qx1y[255]  = {'\0'};
			char filename_qy1y[255]  = {'\0'};
			char filename_z1y[255]   = {'\0'};

			sprintf(filename_eta0,  "%s%c%s", prefix, '-', "scale-coeffs-eta0-mw");
			sprintf(filename_qx0,   "%s%c%s", prefix, '-', "scale-coeffs-qx0-mw");
			sprintf(filename_qy0,   "%s%c%s", prefix, '-', "scale-coeffs-qy0-mw");
			sprintf(filename_z0,    "%s%c%s", prefix, '-', "scale-coeffs-z0-mw");
			sprintf(filename_eta1x, "%s%c%s", prefix, '-', "scale-coeffs-eta1x-mw");
			sprintf(filename_qx1x,  "%s%c%s", prefix, '-', "scale-coeffs-qx1x-mw");
			sprintf(filename_qy1x,  "%s%c%s", prefix, '-', "scale-coeffs-qy1x-mw");
			sprintf(filename_z1x,   "%s%c%s", prefix, '-', "scale-coeffs-z1x-mw");
			sprintf(filename_eta1y, "%s%c%s", prefix, '-', "scale-coeffs-eta1y-mw");
			sprintf(filename_qx1y,  "%s%c%s", prefix, '-', "scale-coeffs-qx1y-mw");
			sprintf(filename_qy1y,  "%s%c%s", prefix, '-', "scale-coeffs-qy1y-mw");
			sprintf(filename_z1y,   "%s%c%s", prefix, '-', "scale-coeffs-z1y-mw");
			
			write_hierarchy_to_file(dirroot, filename_eta0,  this->eta0,  this->levels);
			write_hierarchy_to_file(dirroot, filename_qx0,   this->qx0,   this->levels);
			write_hierarchy_to_file(dirroot, filename_qy0,   this->qy0,   this->levels);
			write_hierarchy_to_file(dirroot, filename_z0,    this->z0,    this->levels);
			write_hierarchy_to_file(dirroot, filename_eta1x, this->eta1x, this->levels);
			write_hierarchy_to_file(dirroot, filename_qx1x,  this->qx1x,  this->levels);
			write_hierarchy_to_file(dirroot, filename_qy1x,  this->qy1x,  this->levels);
			write_hierarchy_to_file(dirroot, filename_z1x,   this->z1x,   this->levels);
			write_hierarchy_to_file(dirroot, filename_eta1y, this->eta1y, this->levels);
			write_hierarchy_to_file(dirroot, filename_qx1y,  this->qx1y,  this->levels);
			write_hierarchy_to_file(dirroot, filename_qy1y,  this->qy1y,  this->levels);
			write_hierarchy_to_file(dirroot, filename_z1y,   this->z1y,   this->levels);
		}
	}

} ScaleCoefficients;